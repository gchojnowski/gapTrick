import os, sys, re


import sys
import tempfile
import numpy as np
import string
import pickle
import time


from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch
import colabfold as cf


from alphafold.common import residue_constants
from alphafold.relax import relax

from alphafold.data import mmcif_parsing
from alphafold.data.templates import (_get_pdb_id_and_chain,
                                      _process_single_hit,
                                      _assess_hhsearch_hit,
                                      _build_query_to_hit_index_mapping,
                                      _extract_template_features,
                                      SingleHitResult,
                                      TEMPLATE_FEATURES)
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.PDB import  PDBParser, MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio import PDB
import io

from pathlib import Path
import pickle
import shutil

from dataclasses import dataclass, replace
from jax.lib import xla_bridge

from optparse import OptionParser, OptionGroup, SUPPRESS_HELP


import iotbx
import iotbx.cif
import iotbx.pdb
from iotbx.pdb import amino_acid_codes as aac

print(xla_bridge.get_backend().platform)

FAKE_MMCIF_HEADER=\
"""data_%(outid)s
#
_entry.id   %(outid)s
_struct_asym.id          A
_struct_asym.entity_id   0
#
_entity_poly.entity_id        0
_entity_poly.type             polypeptide(L)
_entity_poly.pdbx_strand_id   A
#
_entity.id     0
_entity.type   polymer
#
loop_
_chem_comp.id
_chem_comp.type
_chem_comp.name
ALA 'L-peptide linking' ALANINE
ARG 'L-peptide linking' ARGININE
ASN 'L-peptide linking' ASPARAGINE
ASP 'L-peptide linking' 'ASPARTIC ACID'
CYS 'L-peptide linking' CYSTEINE
GLN 'L-peptide linking' GLUTAMINE
GLU 'L-peptide linking' 'GLUTAMIC ACID'
HIS 'L-peptide linking' HISTIDINE
ILE 'L-peptide linking' ISOLEUCINE
LEU 'L-peptide linking' LEUCINE
LYS 'L-peptide linking' LYSINE
MET 'L-peptide linking' METHIONINE
PHE 'L-peptide linking' PHENYLALANINE
PRO 'L-peptide linking' PROLINE
SER 'L-peptide linking' SERINE
THR 'L-peptide linking' THREONINE
TRP 'L-peptide linking' TRYPTOPHAN
TYR 'L-peptide linking' TYROSINE
VAL 'L-peptide linking' VALINE
GLY 'L-peptide linking' GLYCINE
#"""

hhdb_build_template="""
cd %(msa_dir)s
ffindex_build -s ../DB_msa.ff{data,index} .
cd %(hhDB_dir)s
ffindex_apply DB_msa.ff{data,index}  -i DB_a3m.ffindex -d DB_a3m.ffdata  -- hhconsensus -M 50 -maxres 65535 -i stdin -oa3m stdout -v 0
rm DB_msa.ff{data,index}
ffindex_apply DB_a3m.ff{data,index} -i DB_hhm.ffindex -d DB_hhm.ffdata -- hhmake -i stdin -o stdout -v 0
cstranslate -f -x 0.3 -c 4 -I a3m -i DB_a3m -o DB_cs219 
sort -k3 -n -r DB_cs219.ffindex | cut -f1 > sorting.dat

ffindex_order sorting.dat DB_hhm.ff{data,index} DB_hhm_ordered.ff{data,index}
mv DB_hhm_ordered.ffindex DB_hhm.ffindex
mv DB_hhm_ordered.ffdata DB_hhm.ffdata

ffindex_order sorting.dat DB_a3m.ff{data,index} DB_a3m_ordered.ff{data,index}
mv DB_a3m_ordered.ffindex DB_a3m.ffindex
mv DB_a3m_ordered.ffdata DB_a3m.ffdata
cd %(home_path)s
"""

def parse_args():
    """setup program options parsing"""
    parser = OptionParser(usage="Usage: af2_cplx_templates.py [options]", version="0.0.1")


    required_opts = OptionGroup(parser, "Required parameters (model, sequences and a map)")
    parser.add_option_group(required_opts)

    required_opts.add_option("--msa", action="store", \
                            dest="msa", type="string", metavar="FILENAME,FILENAME", \
                  help="comma-separated a3m MSAs. First sequence is a target", default=None)

    required_opts.add_option("--templates", action="store", \
                            dest="templates", type="string", metavar="FILENAME,FILENAME", \
                  help="comma-separated temlates in mmCIF/PDB format", default=None)

    required_opts.add_option("--chain_ids", action="store", \
                            dest="chain_ids", type="string", metavar="CHAR,CHAR", \
                  help="comma-separated template chains corresponding to consequtive MSAs", default=None)

    required_opts.add_option("--cardinality", action="store", dest="cardinality", type="string", metavar="INT,INT", \
                  help="cardinalities of consecutive MSA", default=None)

    required_opts.add_option("--trim", action="store", dest="trim", type="string", metavar="INT,INT", \
                  help="lengths of consecutive target seqs", default=None)

    required_opts.add_option("--num_models", action="store", dest="num_models", type="int", metavar="INT", \
                  help="number of output models (<=5)", default=5)

    required_opts.add_option("--num_recycle", action="store", dest="num_recycle", type="int", metavar="INT", \
                  help="number of recycles", default=3)

    required_opts.add_option("--jobname", action="store", dest="jobname", type="string", metavar="DIRECTORY", \
                  help="output directory name", default=None)

    required_opts.add_option("--nomerge", action="store_true", dest="nomerge", default=False, \
                  help="Use input templates as monomers. Benchmarks only!")

    required_opts.add_option("--data_dir", action="store", \
                            dest="data_dir", type="string", metavar="DIRNAME", \
                  help="AlphaFold2 database", default='/scratch/AlphaFold_DBs/2.3.2')

    required_opts.add_option("--dryrun", action="store_true", dest="dryrun", default=False, \
                  help="check template alignments and quit")

    required_opts.add_option("--relax", action="store_true", dest="relax", default=False, \
                  help="relax top model")


    (options, _args)  = parser.parse_args()
    return (parser, options)

def parse_pdbstring(pdb_string):

    # there may be issues with repeated BREAK lines, that we do not use here anyway
    pdb_string_lines = []
    for _line in pdb_string.splitlines():
        if re.match(r'^BREAK$', _line):
            continue
        pdb_string_lines.append(_line)


    # arghhhh, why do you guys keep changing the interface?
    inp = iotbx.pdb.input(source_info=None, lines=pdb_string_lines)
    try:
        return inp.construct_hierarchy(sort_atoms=False), inp.crystal_symmetry()
    except:
        return inp.construct_hierarchy(), inp.crystal_symmetry()


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

    inputpath=Path(prefix, "input")
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
    plddts,paes,ptmscore= [],[],[]
    unrelaxed_pdb_lines = []
    relaxed_pdb_lines = []
    distograms=[]

    for model_name, params in model_params.items():
        if model_name in use_model:
            print(f"running {model_name}")
            # swap params to avoid recompiling
            # note: models 1,2 have diff number of params compared to models 3,4,5
            if any(str(m) in model_name for m in [1,2]): model_runner = model_runner_1
            if any(str(m) in model_name for m in [3,4,5]): model_runner = model_runner_3
            model_runner.params = params

            processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)


            if 0:#not is_complex:
                input_features = batch_input(
                                        processed_feature_dict,
                                        model_runner,
                                        model_name,
                                        crop_len,
                                        use_templates)
            else:
                input_features = processed_feature_dict


            #prediction_result = model_runner.predict(processed_feature_dict)
            prediction_result = model_runner.predict(input_features, random_seed=random_seed)
            #print("prediction_result.keys()=", prediction_result.keys())
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
            ptmscore.append(prediction_result["ptm"])
            distograms.append(prediction_result["distogram"])

    # rerank models based on pTM (not predicted lddt!)
    #lddt_rank = np.mean(plddts,-1).argsort()[::-1]
    ptm_rank = np.argsort(ptmscore)[::-1]
    output = {}
    print(f"reranking models based on pTM score: {ptm_rank}")
    #print(f"reranking models based on avg. predicted lDDT: {lddt_rank}")
    for n,r in enumerate(ptm_rank):



        print(f"model_{n+1} {np.mean(plddts[r])}")

        with Path(inputpath, f'unrelaxed_model_{n+1}.pdb').open('w') as of: of.write(unrelaxed_pdb_lines[r])

        # relax TOP model only
        if do_relax and n<1:

            pdb_obj = protein.from_pdb_string(unrelaxed_pdb_lines[r])

            print(f"Starting Amber relaxation for model_{n+1}")
            start_time = time.time()

            amber_relaxer = relax.AmberRelaxation(
                                    max_iterations=3,
                                    tolerance=2.39,
                                    stiffness=10.0,
                                    exclude_residues=[],
                                    max_outer_iterations=3,
                                    use_gpu=True)

            _pdb_lines, _, _ = amber_relaxer.process(prot=pdb_obj)

            with Path(inputpath, f'relaxed_model_{n+1}.pdb').open('w') as of: of.write(_pdb_lines)
            print(f"Done, relaxation took {(time.time() - start_time):.1f}s")


        else:
            _pdb_lines = unrelaxed_pdb_lines[r]

        output[n+1] = {"plddt":plddts[r], "pae":paes[r], 'ptm':ptmscore[r], 'pdb':_pdb_lines, 'distogram':distograms[r]}

    return output

def chain2CIF(chain, outid):

    new_ph = iotbx.pdb.hierarchy.root()
    # AF2 expects model.id=1
    new_ph.append_model(iotbx.pdb.hierarchy.model(id="1"))
    new_ph.models()[0].append_chain(chain.detached_copy())
    ogt = aac.one_letter_given_three_letter
    tgo = aac.three_letter_given_one_letter

    poly_seq_block = []
    seq=chain.as_sequence()
    poly_seq_block.append("#")
    poly_seq_block.append("loop_")
    poly_seq_block.append("_entity_poly_seq.entity_id")
    poly_seq_block.append("_entity_poly_seq.num")
    poly_seq_block.append("_entity_poly_seq.mon_id")
    poly_seq_block.append("_entity_poly_seq.hetero")
    for i, aa in enumerate(seq):
        three_letter_aa = tgo[aa]
        poly_seq_block.append(f"0\t{i + 1}\t{three_letter_aa}\tn")

    cif_object = iotbx.cif.model.cif()
    cif_object[outid] = new_ph.as_cif_block()
    cif_object[outid].pop('_chem_comp.id')
    cif_object[outid].pop('_struct_asym.id')

    with io.StringIO() as outstr:
        print(FAKE_MMCIF_HEADER%locals(), file=outstr)
        print("\n".join(poly_seq_block), file=outstr)
        print(cif_object[outid], file=outstr)
        outstr.seek(0)
        return outstr.read()



def template_preps_nomerge(template_fn_list, chain_ids, outpath=None):
    '''
        this one will put requeste chains from each template in a separate AF2-compatible mmCIF
    '''
    converted_template_fns=[]

    selected_chids=chain_ids.split(',')
    idx=0
    for ifn in template_fn_list:

        with open(ifn, 'r') as ifile:
            ph, symm = parse_pdbstring(ifile.read())
            ph.remove_alt_confs(True)

        for chid in selected_chids:

            ph_sel = ph.select(ph.atom_selection_cache().iselection(f"chain {chid} and protein"))

            if not outpath: continue

            outid=f"{idx:04d}"
            converted_template_fns.append(os.path.join(outpath, f"{outid}.cif"))
            with open(converted_template_fns[-1], 'w') as ofile:
                print(chain2CIF(ph_sel.only_chain(), outid), file=ofile)
            idx+=1


    return converted_template_fns


def template_preps(template_fn_list, chain_ids, outpath=None, resi_shift=200):

    converted_template_fns=[]

    selected_chids=chain_ids.split(',')
    idx=0
    for ifn in template_fn_list:
        outid=f"{idx:04d}"
        with open(ifn, 'r') as ifile:
            ph, symm = parse_pdbstring(ifile.read())
            ph.remove_alt_confs(True)

        chaindict={}
        for ch in ph.chains():
            chaindict[ch.id]=ch

        # assembly
        tmp_ph = iotbx.pdb.hierarchy.root()
        tmp_ph.append_model(iotbx.pdb.hierarchy.model(id="0"))
        tmp_ph.models()[0].append_chain(iotbx.pdb.hierarchy.chain(id="A"))

        for ich,chid in enumerate(selected_chids):
            if not len(tmp_ph.only_chain().residue_groups()):
                last_resid = 1
            else:
                last_resid = tmp_ph.only_chain().residue_groups()[-1].resseq_as_int()

            for residx,res in enumerate(chaindict[chid].detached_copy().residue_groups()):
                res.resseq = last_resid+resi_shift+residx
                tmp_ph.only_chain().append_residue_group( res )


        ph_sel = tmp_ph.select(tmp_ph.atom_selection_cache().iselection(f"protein"))

        if not outpath: continue

        converted_template_fns.append(os.path.join(outpath, f"{outid}.cif"))
        with open(converted_template_fns[-1], 'w') as ofile:
            print(chain2CIF(ph_sel.only_chain(), outid), file=ofile)
        idx+=1


    return converted_template_fns

def generate_template_features(query_sequence, db_path, template_fn_list, nomerge=False, dryrun=False):
    home_path=os.getcwd()

    query_seq = SeqRecord(Seq(query_sequence),id="query",name="",description="")

    parent_dir = Path(db_path)
    cif_dir = Path(parent_dir,"mmcif")
    fasta_dir = Path(parent_dir,"fasta")
    hhDB_dir = Path(parent_dir,"hhDB")
    msa_dir = Path(hhDB_dir,"msa")
    db_prefix="DB"

    for dd in [parent_dir, cif_dir, fasta_dir, hhDB_dir, msa_dir]:
        if dd.exists():
            shutil.rmtree(dd)
        dd.mkdir(parents=True)




    template_hit_list=[]
    for template_fn in template_fn_list[:]:
        print(f"Template file: {template_fn}")
        filepath=Path(template_fn)
        with open(filepath, "r") as fh:
            filestr = fh.read()
            mmcif_obj = mmcif_parsing.parse(file_id=filepath.stem,mmcif_string=filestr, catch_all_errors=True)
            mmcif = mmcif_obj.mmcif_object

        if not mmcif: print(mmcif_obj)
        #/cryo_em/AlphaFold/ColabFold/cf_devel/site-packages/alphafold/data/mmcif_parsing.py
        for chain_id,template_sequence in mmcif.chain_to_seqres.items():
            print(chain_id, template_sequence)

            seq_name = filepath.stem.upper()+"_"+chain_id
            seq = SeqRecord(Seq(template_sequence),id=seq_name,name="",description="")

            with  Path(fasta_dir,seq.id+".fasta").open("w") as fh:
                SeqIO.write([seq], fh, "fasta")

        template_seq=seq
        template_seq_path = Path(msa_dir,"template.fasta")
        with template_seq_path.open("w") as fh:
            SeqIO.write([seq], fh, "fasta")

        cmd=hhdb_build_template%locals()
        os.system(cmd)

        hhsearch_runner = hhsearch.HHSearch(binary_path="hhsearch", databases=[hhDB_dir.as_posix()+"/"+db_prefix])
        with io.StringIO() as fh:
            SeqIO.write([query_seq], fh, "fasta")
            seq_fasta = fh.getvalue()

        hhsearch_result = hhsearch_runner.query(seq_fasta)
        hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)

        if len(hhsearch_hits) >0:
            naligned=[]
            for _i,_h in enumerate(hhsearch_hits):
                naligned.append(len(_h.hit_sequence)-_h.hit_sequence.count('-'))
                print()
                print()
                print(f">{_h.name}_{_i+1} coverage is {naligned[-1]} of {len(query_sequence)}")
                print(f"HIT ", _h.hit_sequence)
                print(f"QRY ", _h.query)

            print()
            # in no-merge mode accept multiple alignments, in case target is a homomultimer
            if nomerge:
                for _i,_h in enumerate(hhsearch_hits):
                    if naligned[_i]/len(template_sequence)<0.5: continue
                    print(f' --> Selected alignment #{_i+1}')
                    template_hit_list.append([mmcif,_h])
            else:
                print(f' --> Selected alignment #{np.argmax(naligned)+1}')
                hit = hhsearch_hits[np.argmax(naligned)]
                hit = replace(hit,**{"name":template_seq.id})

                template_hit_list.append([mmcif, hit])
        print()

    print()
    if dryrun: exit(1)

    template_features = {}
    for template_feature_name in TEMPLATE_FEATURES:
      template_features[template_feature_name] = []

    for mmcif,hit in template_hit_list:

        hit_pdb_code, hit_chain_id = _get_pdb_id_and_chain(hit)
        mapping = _build_query_to_hit_index_mapping(hit.query, hit.hit_sequence, hit.indices_hit, hit.indices_query,query_sequence)
        print(">"+hit.name)
        print("hit ", hit.hit_sequence)
        print("qry ", hit.query)

        template_sequence = hit.hit_sequence.replace('-', '')

        features, realign_warning = _extract_template_features(
            mmcif_object=mmcif,
            pdb_id=hit_pdb_code,
            mapping=mapping,
            template_sequence=template_sequence,
            query_sequence=query_sequence,
            template_chain_id=hit_chain_id,
            kalign_binary_path="kalign")

        features['template_sum_probs'] = [hit.sum_probs]

        single_hit_result = SingleHitResult(features=features, error=None, warning=None)

        for k in template_features:
            if isinstance(template_features[k], (np.ndarray, np.generic) ):
                template_features[k] = np.append(template_features[k], features[k])
            else:
                template_features[k].append(features[k])

        for name in template_features:
            template_features[name] = np.stack(template_features[name], axis=0).astype(TEMPLATE_FEATURES[name])

    for key,value in template_features.items():
        if np.all(value==0): print("ERROR: Some template features are empty")

    return template_features

def combine_msas(query_sequences, input_msas, query_cardinality, query_trim):
    pos=0
    #msa_combined=[">query\n"+query_seq_combined]
    msa_combined=[]
    #_blank_seq = [ ("-" * len(seq[query_trim[n][0]:query_trim[n][1]])) for n, seq in enumerate(query_sequences) for _ in range(query_cardinality[n]) ]
    _blank_seq = [ ("-" * len(seq)) for n, seq in enumerate(query_sequences) for _ in range(query_cardinality[n]) ]

    for n, seq in enumerate(query_sequences):
        for j in range(0, query_cardinality[n]):
            for _desc, _seq in zip(input_msas[n].descriptions, input_msas[n].sequences[:]):
                if not len(set(_seq[query_trim[n][0]:query_trim[n][1]]))>1: continue
                msa_combined.append(">%s"%_desc)
                msa_combined.append("".join(_blank_seq[:pos] + [re.sub('[a-z]', '', _seq)[query_trim[n][0]:query_trim[n][1]]] + _blank_seq[pos + 1 :]))
            pos += 1


    msas=[pipeline.parsers.parse_a3m("\n".join(msa_combined))]

    for _i, _m in enumerate(msas):
        print(_i, '\n', "\n".join(_m.sequences[:1]))

    return msas





def runme(msa_filenames,
          query_cardinality =   [1,0],
          query_trim        =   [[0,10000],[0,10000]],
          template_fn_list  =   None,
          num_models        =   1,
          jobname           =   'test',
          data_dir          =   '/scratch/AlphaFold_DBs/2.3.2',
          num_recycle       =   3,
          chain_ids         =   None,
          dryrun            =   False,
          do_relax          =   False,
          nomerge           =   False):

    msas=[]
    for a3m_fn in msa_filenames:
        with open(a3m_fn, 'r') as fin:
            msas.append(pipeline.parsers.parse_a3m(fin.read()))


    query_sequences=[_m.sequences[0][query_trim[_i][0]:query_trim[_i][1]] for _i,_m in enumerate(msas)]# for _ in range(query_cardinality[_i])]
    query_seq_extended=[_m.sequences[0][query_trim[_i][0]:query_trim[_i][1]] for _i,_m in enumerate(msas) for _ in range(query_cardinality[_i])]
    query_seq_combined="".join(query_seq_extended)


    msas = combine_msas(query_sequences, msas, query_cardinality, query_trim)



    #reproduce af2-like output paths
    # do not clean jobpath - processed template will be stored there before job is started
    jobpath=Path(jobname)
    inputpath=Path(jobname, "input")
    msaspath=Path(jobname, "input", "msas", "A")
    for dd in [inputpath, msaspath]:
        if dd.exists():
            shutil.rmtree(dd)
        dd.mkdir(parents=True)

    # query sequence
    with Path(jobpath, 'input.fasta').open('w') as of:
        of.write(">input\n%s\n"%query_seq_combined)
    # a3m
    a3m_fn='input_combined.a3m'
    with Path(msaspath, a3m_fn).open('w') as of:
        for _i, _m in enumerate(msas):
            of.write("\n".join([">%s\n%s"%(_d,_s) for (_d,_s) in zip(_m.descriptions,_m.sequences)]))

    print(query_seq_combined)
    print()
    if nomerge:
        template_fn_list = template_preps_nomerge(template_fn_list, chain_ids, outpath=inputpath)
    else:
        template_fn_list = template_preps(template_fn_list, chain_ids, outpath=inputpath)

    with tempfile.TemporaryDirectory() as tmp_path:
        print("Created tmp path ", tmp_path)
        template_features = generate_template_features(query_sequence   =   query_seq_combined,
                                                       db_path          =   tmp_path,
                                                       template_fn_list =   template_fn_list,
                                                       nomerge          =   nomerge,
                                                       dryrun           =   dryrun)

    use_model = {}
    model_params = {}
    model_runner_1 = None
    model_runner_3 = None
    for model_idx in range(1,6)[:num_models]:
        model_name=f"model_{model_idx}"
        use_model[model_name] = True
        if model_name not in list(model_params.keys()):
            model_params[model_name] = data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir=data_dir)
            if model_idx == 1:
                model_config = config.model_config(model_name+"_ptm")
                model_config.data.common.num_recycle = num_recycle
                model_config.model.num_recycle = num_recycle
                model_config.data.eval.num_ensemble = 1
                model_runner_1 = model.RunModel(model_config, model_params[model_name])
            if model_idx == 3:
                model_config = config.model_config(model_name+"_ptm")
                model_config.data.common.num_recycle = num_recycle
                model_config.model.num_recycle = num_recycle
                model_config.data.eval.num_ensemble = 1
                model_runner_3 = model.RunModel(model_config, model_params[model_name])

    is_complex=True


    # gather features
    feature_dict = {
        **pipeline.make_sequence_features(sequence=query_seq_combined, description="none", num_res=len(query_seq_combined)),
        **pipeline.make_msa_features(msas=msas),
        **template_features
    }

    feature_dict["asym_id"] = \
            np.array( [int(n) for n, l in enumerate(tuple(map(len, query_seq_extended))) for _ in range(0, l)] )

    feature_dict['assembly_num_chains']=len(query_seq_extended)

    output = predict_structure(jobname, query_seq_combined, feature_dict,
                             Ls=tuple(map(len, query_seq_extended)),
                             model_params=model_params, use_model=use_model,
                             model_runner_1=model_runner_1,
                             model_runner_3=model_runner_3,
                             is_complex=is_complex,
                             do_relax=do_relax)


    with Path(inputpath, 'features.pkl').open('wb') as of: pickle.dump(feature_dict, of, protocol=pickle.HIGHEST_PROTOCOL)
    # pickels for models (plddt-ranked)
    for idx,dat in output.items():

        pdb_fn = 'ranked_%d.pdb'%idx
        print(f"{pdb_fn} <pLDDT>={np.mean(dat['plddt']):6.4f} pTM={dat['ptm']:6.4f}")
        with Path(inputpath, pdb_fn).open('w') as of: of.write(dat['pdb'])

        outdict={'predicted_aligned_error':dat['pae'], 'ptm':dat['ptm'], 'plddt':dat['plddt'], 'distogram':dat['distogram']}
        pkl_fn = 'result_model_%d_ptm.pkl'%(idx+1)
        with Path(inputpath, pkl_fn).open('wb') as of: pickle.dump(outdict, of, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    #esx_tests_ddce_with_template()

    (parser, options) = parse_args()

    print( " ==> Command line: af2_cplx_templates.py %s" % (" ".join(sys.argv[1:])) )

    msas = options.msa.split(',')

    if not options.trim:
        trim = [[0,9999]]*len(msas)
    else:
        #trim = tuple(map(int,options.trim.split(',')))
        trim=[tuple(map(int, _.split(":"))) for _ in options.trim.split(",")]
    print("TRIM: ", trim)
    if not options.cardinality:
        cardinality = [1]*len(msas)
    else:
        cardinality = tuple(map(int,options.cardinality.split(',')))

    if options.jobname is None:
        print('Define jobname - output directory')
        exit(0)

    runme(msa_filenames     =   msas,
          query_cardinality =   cardinality,
          query_trim        =   trim,
          num_models        =   options.num_models,
          template_fn_list  =   options.templates.split(',') if options.templates else None,
          jobname           =   options.jobname,
          data_dir          =   options.data_dir,
          num_recycle       =   options.num_recycle,
          chain_ids         =   options.chain_ids,
          dryrun            =   options.dryrun,
          do_relax          =   options.relax,
          nomerge           =   options.nomerge)




if __name__=="__main__":
    main()
