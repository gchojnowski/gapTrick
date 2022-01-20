import os, sys, re


import sys
import numpy as np
import string
import pickle
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch
import colabfold as cf

from alphafold.data import mmcif_parsing
from alphafold.data.templates import (_get_pdb_id_and_chain,
                                                                            _process_single_hit,
                                                                            _assess_hhsearch_hit,
                                                                            _build_query_to_hit_index_mapping,
                                                                            _extract_template_features,
                                                                            SingleHitResult,
                                                                            TEMPLATE_FEATURES)

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

def mk_mock_template(query_sequence):
  # since alphafold's model requires a template input
  # we create a blank example w/ zero input, confidence -1
  ln = len(query_sequence)
  output_templates_sequence = "-"*ln
  output_confidence_scores = np.full(ln,-1)
  templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
  templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
  templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                    templates.residue_constants.HHBLITS_AA_TO_ID)
  template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
                       'template_all_atom_masks': templates_all_atom_masks[None],
                       'template_sequence': [f'none'.encode()],
                       'template_aatype': np.array(templates_aatype)[None],
                       'template_confidence_scores': output_confidence_scores[None],
                       'template_domain_names': [f'none'.encode()],
                       'template_release_date': [f'none'.encode()]}
  return template_features

def mk_template(a3m_lines, template_paths):
  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=template_paths,
      max_template_date="2100-01-01",
      max_hits=20,
      kalign_binary_path="kalign",
      release_dates_path=None,
      obsolete_pdbs_path=None)

  hhsearch_pdb70_runner = hhsearch.HHSearch(binary_path="hhsearch", databases=[f"{template_paths}/pdb70"])

  hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
  hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
  templates_result = template_featurizer.get_templates(query_sequence=query_sequence,
                                                       query_pdb_code=None,
                                                       query_release_date=None,
                                                       hits=hhsearch_hits)
  return templates_result.features

def pad_sequences(a3m_lines, query_sequences, query_cardinality):
    _blank_seq = [("-" * len(seq)) for n, seq in enumerate(query_sequences) for _ in range(query_cardinality[n])]
    a3m_lines_combined = []
    pos = 0
    for n, seq in enumerate(query_sequences):
        for j in range(0, query_cardinality[n]):
            lines = a3m_lines[n].split("\n")
            for a3m_line in lines:
                if len(a3m_line) == 0: continue
                if a3m_line.startswith(">"):
                    a3m_lines_combined.append(a3m_line)
                else:
                    a3m_lines_combined.append("".join(_blank_seq[:pos] + [a3m_line] + _blank_seq[pos + 1 :]))
            pos += 1
    return "\n".join(a3m_lines_combined)


def set_bfactor(pdb_filename, bfac, idx_res, chains):
  I = open(pdb_filename,"r").readlines()
  O = open(pdb_filename,"w")
  residx=0
  for line in I:
    if line[0:6] == "ATOM  ":
      seq_id = int(line[22:26].strip()) - 1
      seq_id = np.where(idx_res == seq_id)[0][0]
      O.write(f"{line[:21]}{chains[residx]}{line[22:60]}{bfac[residx]:6.2f}{line[66:]}")
      residx+=1
  O.close()


def predict_structure(prefix,
                      query_sequence,
                      feature_dict,
                      Ls,
                      model_params,
                      use_model,
                      model_runner_1,
                      model_runner_3,
                      do_relax=False,
                      random_seed=0, 
                      is_complex=False):

    """Predicts structure using AlphaFold for the given sequence."""

    seq_len = len(query_sequence)

    # Minkyung's code
    # add big enough number to residue index to indicate chain breaks
    idx_res = feature_dict['residue_index']
    L_prev = 0
    # Ls: number of residues in each chain
    for L_i in Ls[:-1]:
        idx_res[L_prev+L_i:] += 200
        L_prev += L_i
    chains = list("".join([string.ascii_uppercase[n]*L for n,L in enumerate(Ls)]))
    feature_dict['residue_index'] = idx_res
    #print(idx_res)
    #print(chains)
    # Run the models.
    plddts,paes = [],[]
    unrelaxed_pdb_lines = []
    relaxed_pdb_lines = []

    for model_name, params in model_params.items():
        if model_name in use_model:
            print(f"running {model_name}")
            # swap params to avoid recompiling
            # note: models 1,2 have diff number of params compared to models 3,4,5
            if any(str(m) in model_name for m in [1,2]): model_runner = model_runner_1
            if any(str(m) in model_name for m in [3,4,5]): model_runner = model_runner_3
            model_runner.params = params

            processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)


            if not is_complex:
                input_features = batch_input(
                                        processed_feature_dict,
                                        model_runner,
                                        model_name,
                                        crop_len,
                                        use_templates)
            else:
                input_features = processed_feature_dict


            #prediction_result = model_runner.predict(processed_feature_dict)
            prediction_result, recycles = model_runner.predict(input_features)

            print(len(prediction_result["plddt"]), seq_len)
            mean_plddt = np.mean(prediction_result["plddt"][:seq_len])
            mean_ptm = prediction_result["ptm"]

            final_atom_mask = prediction_result["structure_module"]["final_atom_mask"]
            b_factors = prediction_result["plddt"][:, None] * final_atom_mask

            model_type = "AlphaFold2-ptm"
            if is_complex and model_type == "AlphaFold2-ptm":
                resid2chain = {}
                input_features["asym_id"] = feature_dict["asym_id"]
                input_features["aatype"] = input_features["aatype"][0]
                input_features["residue_index"] = input_features["residue_index"][0]
                curr_residue_index = 1
                res_index_array = input_features["residue_index"].copy()
                res_index_array[0] = 0
                for i in range(1, input_features["aatype"].shape[0]):
                    if (input_features["residue_index"][i] - input_features["residue_index"][i - 1]) > 1:
                        curr_residue_index = 0

                    res_index_array[i] = curr_residue_index
                    curr_residue_index += 1

                input_features["residue_index"] = res_index_array

            unrelaxed_protein = protein.from_prediction(
                                                features=input_features,
                                                result=prediction_result,
                                                b_factors=b_factors,
                                                remove_leading_feature_dimension=not is_complex)

            #unrelaxed_protein = protein.from_prediction(processed_feature_dict,prediction_result)
            unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))

            #plddts.append(prediction_result['plddt'])
            paes.append(prediction_result['predicted_aligned_error'])
            plddts.append(prediction_result["plddt"][:seq_len])
            #ptmscore.append(prediction_result["ptm"])

    # rerank models based on predicted lddt
    lddt_rank = np.mean(plddts,-1).argsort()[::-1]
    out = {}
    print("reranking models based on avg. predicted lDDT")
    for n,r in enumerate(lddt_rank):
        print(plddts[r])
        print(f"model_{n+1} {np.mean(plddts[r])}")

        unrelaxed_pdb_path = f'{prefix}_unrelaxed_model_{n+1}.pdb'    
        with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines[r])
        #set_bfactor(unrelaxed_pdb_path, plddts[r], idx_res, chains)

        if do_relax:
            relaxed_pdb_path = f'{prefix}_relaxed_model_{n+1}.pdb'
            with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_lines[r])
            #set_bfactor(relaxed_pdb_path, plddts[r], idx_res, chains)

        out[f"model_{n+1}"] = {"plddt":plddts[r], "pae":paes[r]}

    return out





if 0:
    model_name="model_1"
    _=data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir="/cryo_em/AlphaFold/DBs")
    print(dir(_))

homooligomer=2
num_models=1
query_cardinality=[2,0]
query_trim=[10000,10000]

piaq_a3m='/home/gchojnowski/af2_jupyter/PIAQ_test_af2mmer/input/msas/bfd_uniclust_hits_dimer.a3m'
piaq_a3m='/home/gchojnowski/af2_jupyter/PIAQ_test_cf_mono/1.a3m'
piaa_a3m='/home/gchojnowski/af2_jupyter/PIAQAAA_test_cf_mono/1.a3m'

msas=[]
for a3m_fn in [piaq_a3m, piaa_a3m]:
    with open(a3m_fn, 'r') as fin:
        msas.append(pipeline.parsers.parse_a3m(fin.read()))

#msas = [msa]#*homooligomer
query_sequences=[_m.sequences[0][:query_trim[_i]] for _i,_m in enumerate(msas)]# for _ in range(query_cardinality[_i])]
#query_sequence=msa.sequences[0]
query_seq_extended=[_m.sequences[0][:query_trim[_i]] for _i,_m in enumerate(msas) for _ in range(query_cardinality[_i])]
query_seq_combined="".join(query_seq_extended)
jobname='piaqpiaa_test'


template_features = mk_mock_template(query_seq_combined)


use_model = {}
model_params = {}
model_runner_1 = None
model_runner_3 = None
for model_name in ["model_1","model_2","model_3","model_4","model_5"][:num_models]:
    use_model[model_name] = True
    if model_name not in list(model_params.keys()):
        model_params[model_name] = data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir="/cryo_em/AlphaFold/DBs")
        if model_name == "model_1":
            model_config = config.model_config(model_name+"_ptm")
            model_config.data.eval.num_ensemble = 1
            model_runner_1 = model.RunModel(model_config, model_params[model_name])
        if model_name == "model_3":
            model_config = config.model_config(model_name+"_ptm")
            model_config.data.eval.num_ensemble = 1
            model_runner_3 = model.RunModel(model_config, model_params[model_name])

is_complex=False

if sum(query_cardinality) > 1:
    # make multiple copies of msa for each copy
    # AAA------
    # ---AAA---
    # ------AAA
    #
    # note: if you concat the sequences (as below), it does NOT work
    # AAAAAAAAA
    #a3m_lines_combined = pad_sequences(a3m_lines            =   [_.sequences for _ in msas],
    #                                   query_sequences      =   query_sequences,
    #                                   query_cardinality    =   [1]*homooligomer)

    #msas = [pipeline.parsers.parse_a3m(a3m_lines_combined)]
    #msas=[]
    pos=0
    msa_combined=[">query\n"+query_seq_combined]
    _blank_seq = [ ("-" * len(seq[:query_trim[n]])) for n, seq in enumerate(query_sequences) for _ in range(query_cardinality[n]) ]
    for n, seq in enumerate(query_sequences):
        for j in range(0, query_cardinality[n]):
            for _desc, _seq in zip(msas[n].descriptions, msas[n].sequences[:2]):
                if not len(set(_seq[:query_trim[n]]))>1: continue
                msa_combined.append(">%s"%_desc)
                msa_combined.append("".join(_blank_seq[:pos] + [re.sub('[a-z]', '', _seq)[:query_trim[n]]] + _blank_seq[pos + 1 :]))
            pos += 1


    msas=[pipeline.parsers.parse_a3m("\n".join(msa_combined))]

    #Ln = tuple(map(len,query_sequences))
    #a3m_lines=[">101\n"+query_seq_combined]
    #for o in range(len(query_sequences)):
    #    L = sum(Ln[:o])
    #    R = sum(Ln[(o+1):])# * (homooligomer-(o+1))
    #    for _d, _s in zip(msas[o].descriptions, msas[o].sequences[:2]):
    #        a3m_lines.append(">"+_d)
    #        a3m_lines.append("-"*L+_s+"-"*R)
    #    #msas.append(["-"*L+seq+"-"*R for seq in msa.sequences])
    #    #msas.append(pipeline.parsers.parse_a3m(a3m_lines))
    #    #deletion_matrices.append([[0]*L+mtx+[0]*R for mtx in deletion_matrix])

    #msas=[pipeline.parsers.parse_a3m("\n".join(a3m_lines))]
    #for _i, _m in enumerate(msas):
    #    print(_i, '\n', "\n".join(_m.sequences[:10]))
    #print(msas[0].sequences[0])
    #exit(1)
    is_complex=True

# gather features
feature_dict = {
        **pipeline.make_sequence_features(sequence=query_seq_combined, description="none", num_res=len(query_seq_combined)),
        **pipeline.make_msa_features(msas=msas),
        **template_features
}
feature_dict["asym_id"] = np.array( [int(n) for n, l in enumerate(tuple(map(len, query_seq_extended))) for _ in range(0, l)] )

outs = predict_structure(jobname, query_seq_combined, feature_dict,
                       Ls=tuple(map(len, query_seq_extended)),
                       model_params=model_params, use_model=use_model,
                       model_runner_1=model_runner_1,
                       model_runner_3=model_runner_3,
                       is_complex=is_complex,
                       do_relax=False)

