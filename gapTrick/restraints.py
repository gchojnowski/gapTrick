
__author__ = "Grzegorz Chojnowski"
__date__ = "5 May 2026"

import os, sys, re, io
import json
from pathlib import Path
import glob
import os
import pickle
import numpy as np
import string

from gapTrick.pdb_utils import parse_pdb_bio, get_prot_chains_bio, ogt, tgo

# templates and dicts for restrain generator
## all 1-3 atom pairs incl side chains
sc_restraints="""ARG:N-CG,N-CD,N-NE,N-CZ,CA-CD,CA-NE,CA-CZ,CB-NE,CB-CZ
ASN:N-CG,N-OD1,N-ND2,CA-OD1,CA-ND2
ASP:N-CG,N-OD1,N-OD2,CA-OD1,CA-OD2
CYS:N-SG
GLN:N-CG,N-CD,N-OE1,N-NE2,CA-CD,CA-OE1,CA-NE2,CB-OE1,CB-NE2
GLU:N-CG,N-CD,N-OE1,N-OE2,CA-CD,CA-OE1,CA-OE2,CB-OE1,CB-OE2
HIS:N-CG,N-ND1,N-CD2,N-CE1,N-NE2,CA-ND1,CA-CD2,CA-CE1,CA-NE2,CB-CE1,CB-NE2
ILE:N-CG1,N-CG2,N-CD1,CA-CD1
LEU:N-CG,N-CD1,N-CD2,CA-CD1,CA-CD2
LYS:N-CG,N-CD,N-CE,N-NZ,CA-CD,CA-CE,CA-NZ,CB-CE,CB-NZ
MET:N-CG,N-SD,N-CE,CA-SD,CA-CE,CB-CE
PHE:N-CG,N-CD1,N-CD2,N-CE1,N-CE2,N-CZ,CA-CD1,CA-CD2,CA-CE1,CA-CE2,CA-CZ,CB-CE1,CB-CE2,CB-CZ
SER:N-OG
THR:N-OG1,N-CG2
TRP:N-CG,N-CD1,N-CD2,N-NE1,N-CE2,N-CE3,N-CZ2,N-CZ3,CA-CD1,CA-CD2,CA-NE1,CA-CE2,CA-CE3,CA-CZ2,CA-CZ3,CB-NE1,CB-CE2,CB-CE3,CB-CZ2,CB-CZ3
TYR:N-CG,N-CD1,N-CD2,N-CE1,N-CE2,N-CZ,CA-CD1,CA-CD2,CA-CE1,CA-CE2,CA-CZ,CB-CE1,CB-CE2,CB-CZ
VAL:N-CG1,N-CG2"""

sc_restraints=dict([(_aa.split(":")[0], [_.split('-') for _ in _aa.split(":")[1].split(',')]) for _aa in sc_restraints.splitlines()])

## generic refmac/coot distance restrain template
refmac_dist_generic="""\
exte dist first chain %(A_chain)s resi %(A_resid)s ins %(A_inscode)s atom %(A_atom_name)s second chain %(B_chain)s resi %(B_resid)s ins %(B_inscode)s atom %(B_atom_name)s value %(mean)f sigma %(sigma)f type 1"""



def make_restraint_scripts(prefix, feature_dict, logger, distance_cutoff=8.0):

    datadir=Path(prefix)
    datadict = {}

    for fn in glob.glob("%s/output/result*.pkl" % datadir):
        with open(fn, 'rb') as ifile:
            data = pickle.load(ifile)
        datadict[fn]=data

    for rank,k in enumerate(sorted(datadict, key=lambda x:datadict[x]['ptm'], reverse=True)):
        datadict[k]['rank']=rank+1

    topmodel_fn=None
    for _fn in datadict:
        if datadict[_fn]['rank']==1:
            topmodel_fn = _fn
            break

    predicted_distogram = datadict[topmodel_fn].get('distogram', None)
    if predicted_distogram is None: return None

    # parse residue ids mappings between input, template (merged), and predicted model
    with open(Path(datadir, "input", "mappings.json"), "r") as ifile:
        mappings = json.load(ifile)

    tpl_fn,residx_mappings_t2i =  list(mappings['template2input_mapping'].items())[0]
    tpl_fn,residx_mappings_m2t =  list(mappings['model2template_mappings'].items())[0]


    structure = parse_pdb_bio(Path(datadir, "output", "ranked_0.pdb"), outid="XYZ", remove_alt_confs=True)
    model_pred = get_prot_chains_bio(structure, logger)
    chain_seq_dict = {}
    for chain in model_pred:
        chain_seq_dict[chain.id]="".join([ogt[_r.get_resname()] for _r in chain.get_unpacked_list()])

    structure = parse_pdb_bio(Path(datadir, "input", "0000_inp.cif"), outid="XYZ", remove_alt_confs=True)
    model_inp = get_prot_chains_bio(structure, logger)

    structure = parse_pdb_bio(Path(datadir, "input", "0000.cif"), outid="XYZ", remove_alt_confs=True)
    model_tpl = get_prot_chains_bio(structure, logger)

    predicted_distogram = datadict[topmodel_fn].get('distogram', None)

    #probs = softmax(predicted_distogram['logits'], axis=-1)
    x = predicted_distogram['logits']
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    probs = exp_x_shifted / np.sum(exp_x_shifted, axis=-1, keepdims=True)


    bin_edges = predicted_distogram['bin_edges']


    # chainid mapping helper for AF2-muiltimer
    asym_id = feature_dict['asym_id']
    assembly_num_chains = feature_dict['assembly_num_chains']

    # for compatibility with versions pre 0.3.8 (previously parsed single chain preds only!)
    if assembly_num_chains is None:
        assembly_num_chains = 1
        asym_id = [1]*len(datadict[topmodel_fn]['plddt'])

    bin_idx=np.max(np.where(bin_edges<distance_cutoff))
    below8pbty = np.sum(probs, axis=2, where=(np.arange(probs.shape[-1])<bin_idx))

    chain_ids = string.ascii_uppercase
    chain_lens = []
    for i in range(assembly_num_chains):
        chain_lens.append(np.sum(np.array(asym_id)==(i+1)))

    chain_lens = np.array(chain_lens)


    restraints_model=[]
    restraints_input=[]

    # generate SC restraints
    input_dict = {}
    for chain in model_inp:
        _c = input_dict.setdefault(chain.id, {})
        for residue in chain:
            _c[residue.get_id()[1]] = residue

    model_dict = {}
    for chain in model_pred:
        _c = model_dict.setdefault(chain.id, {})
        for residue in chain:
            _c[residue.get_id()[1]] = residue

    template_resi_list = []
    for chain in model_tpl:
        for residue in chain:
            template_resi_list.append(residue)

    for i in range(len(asym_id)):

        ci = int(asym_id[i]-1)
        model_chain = chain_ids[ci]
        model_resid = 1+i-sum(chain_lens[:ci])

        model_residue = model_dict[model_chain][model_resid]
        model_atom_dict = dict([(_.get_name(), _.get_coord()) for _ in model_residue.get_atoms()])

        d={}
        d['A_resid'] = d['B_resid'] = model_resid
        d['A_inscode'] = d['B_inscode'] = '.'
        d['A_chain'] = d['B_chain'] = model_chain
        for _aa in sc_restraints.get(model_residue.get_resname(), []):

            d['A_atom_name']=_aa[0]
            d['B_atom_name']=_aa[1]

            d['mean'] = np.linalg.norm(model_atom_dict[_aa[0]] - model_atom_dict[_aa[1]])
            d['sigma'] = 0.5
            restraints_model.append(refmac_dist_generic%d)

        # now, generate sc restraints for input structure
        # there is a tricky residue id mapping that needs to be considered:
        # - restraints to distances in prediciton
        # - skip gaps in input (prediction has no gaps)
        # - skip truncated atoms in input-model SCs (prediction is always complete)

        try:
            resid_tpl = template_resi_list[residx_mappings_m2t[str(i)]].get_id()[1]
            input_chain, input_resid = residx_mappings_t2i[str(resid_tpl)]
        except:
            continue

        d={}
        d['A_resid'] = d['B_resid'] = input_resid
        d['A_inscode'] = d['B_inscode'] = '.'
        d['A_chain'] = d['B_chain'] = input_chain


        input_residue = input_dict[input_chain][input_resid]

        input_atom_dict = dict([(_.get_name(), _.get_coord()) for _ in input_residue.get_atoms()])


        #print(i+1, input_chain, input_resid, input_residue.get_resname(), model_chain, model_resid, model_residue.get_resname() )

        assert model_residue.get_resname() == input_residue.get_resname()

        for _aa in sc_restraints.get(model_residue.get_resname(), []):

            if not (_aa[0] in input_atom_dict.keys() and _aa[1] in input_atom_dict.keys()): continue

            d['A_atom_name']=_aa[0]
            d['B_atom_name']=_aa[1]

            d['mean'] = np.linalg.norm(model_atom_dict[_aa[0]] - model_atom_dict[_aa[1]])
            d['sigma'] = 0.5
            restraints_input.append(refmac_dist_generic%d)

    no_sc_restraints_input = len(restraints_input)
    no_sc_restraints_model = len(restraints_model)


    # generate long-range BB restraints for residue pairs that are likely to be at the distance_cutoff from each other
    for idx, (i,j) in enumerate(zip(*np.where(below8pbty>0.79))):
        d = {}

        ci = int(asym_id[i]-1)
        cj = int(asym_id[j]-1)

        # skipp diag
        if i==j: continue

        resi = 1+i-sum(chain_lens[:ci])
        resj = 1+j-sum(chain_lens[:cj])
        resni = tgo[chain_seq_dict[chain_ids[ci]][int(resi)-1]].upper()
        resnj = tgo[chain_seq_dict[chain_ids[cj]][int(resj)-1]].upper()

        mean = np.sum(probs[i,j,1:-1] * bin_edges[:-1])
        sd = np.sum(probs[i,j,1:-1] * (bin_edges[:-1]-mean)**2)



        d['A_resid'] = resi
        d['A_inscode'] = '.'
        d['A_chain'] = chain_ids[ci]
        d['A_atom_name']='CA' if resni=="GLY" else "CB"
        d['B_resid'] = resj
        d['B_inscode'] = '.'
        d['B_chain'] = chain_ids[cj]
        d['B_atom_name']='CA' if resnj=="GLY" else "CB"
        d['mean'] = mean
        d['sigma'] = sd

        restraints_model.append(refmac_dist_generic%d)
        #print(f"{'*' if ci!=cj else ' '} {resi:-4d}/{chain_ids[ci]}/{resni} {resj:-4d}/{chain_ids[cj]}/{resnj} {mean:5.2f} {sd:5.2f} i={i} j={j}")

        try:
            A_resid_tpl = template_resi_list[residx_mappings_m2t[str(i)]].get_id()[1]
            A_input_chain, A_input_resid = residx_mappings_t2i[str(A_resid_tpl)]
            B_resid_tpl = template_resi_list[residx_mappings_m2t[str(j)]].get_id()[1]
            B_input_chain, B_input_resid = residx_mappings_t2i[str(B_resid_tpl)]
        except:
            continue


        d['A_resid'] = A_input_resid
        d['A_chain'] = A_input_chain
        d['B_resid'] = B_input_resid
        d['B_chain'] = B_input_chain
        restraints_input.append(refmac_dist_generic%d)


    logger.info("\n\n")
    logger.info(f"Generated {no_sc_restraints_input}/{len(restraints_input)-no_sc_restraints_input} sc/mc distance restraints for the input model")
    logger.info(f"Generated {no_sc_restraints_model}/{len(restraints_model)-no_sc_restraints_model} sc/mc distance restraints for the predicted model")
    logger.info("\n\n")

    with open(Path(datadir, "output", "model_refmac_restraints.txt"), "w") as ofile:
        ofile.write("\n".join(restraints_model))

    with open(Path(datadir, "output", "input_refmac_restraints.txt"), "w") as ofile:
        ofile.write("\n".join(restraints_input))
