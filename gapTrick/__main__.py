import os, sys, re, io
import subprocess

import uuid
import glob

import sys
import tempfile
import numpy as np
import jax.numpy as jnp
import string
import pickle
import time
import json
from itertools import groupby
from operator import itemgetter

import requests
import tarfile
from datetime import datetime

from pathlib import Path
import pickle
import shutil

MMSEQS_API_SERVER = "https://api.colabfold.com"
MMSEQS_API_SERVER = "https://a3m.mmseqs.com"

from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch
#import colabfold as cf


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
#from Bio import pairwise2
from Bio import Align
from Bio.PDB import PDBIO, PDBParser, Superimposer, MMCIFParser, Select
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.vectors import rotaxis2m
from Bio.PDB.vectors import Vector

from dataclasses import dataclass, replace
#from jax.lib import xla_bridge

from optparse import OptionParser, OptionGroup, SUPPRESS_HELP
import random

import logging
import logging.config
logging.config.dictConfig({'version': 1,'disable_existing_loggers': True,})
logger = logging.getLogger(__name__)

# try to import a plotter lib and disable plotting if not available
# eg due to missing matplolib (not installed with AlphaFold by default)
try:
    rootpath = Path( __file__ ).parent.absolute()
    sys.path.append(os.path.join(rootpath, '..', 'af2plots'))
    sys.path.append(os.path.join(rootpath))
    from af2plots.plotter import plotter
    PLOTTER_AVAILABLE = 1
except:
    logger.info("WARNING: cannot initiate figure plotter")
    PLOTTER_AVAILABLE = 0
try:
    from gapTrick import version
except:
    import version

tgo = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'O': 'PYL', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'U': 'SEC', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR', 'X': 'UNK'}
ogt = dict([(tgo[_k], _k) for _k in tgo])

#logger.info(xla_bridge.get_backend().platform)

# templates for a pymol script visualising predficted contacts
pymol_dist_generic="""\
dist \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s and name \"%(A_atom_name)s\" and alt \'\', \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s and name \"%(B_atom_name)s\" and alt \'\'"""

pymol_header=f"load %(modelid)s.pdb\nshow_as cartoon, %(modelid)s\nset label_size, 0\nutil.cbc %(modelid)s"

chimerax_footer="distance style radius 0.15\ndistance style color red\ndistance style dashes 0\ncolor bychain"
chimerax_dist_generic=\
        "\n".join(["distance #$1/%(A_chain)s:%(A_resid)s@%(A_atom_name)s #$1/%(B_chain)s:%(B_resid)s@%(B_atom_name)s",
                   "show #$1/%(A_chain)s:%(A_resid)s bonds",
                   "show #$1/%(B_chain)s:%(B_resid)s bonds"])

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
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.data_content_type
_pdbx_audit_revision_history.major_revision
_pdbx_audit_revision_history.minor_revision
_pdbx_audit_revision_history.revision_date
1 'Structure model' 1 0 1878-05-14
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


MMCIF_ATOM_BLOCK_HEADER=\
"""loop_
   _atom_site.group_PDB
   _atom_site.id
   _atom_site.label_atom_id
   _atom_site.label_alt_id
   _atom_site.label_comp_id
   _atom_site.auth_asym_id
   _atom_site.auth_seq_id
   _atom_site.pdbx_PDB_ins_code
   _atom_site.Cartn_x
   _atom_site.Cartn_y
   _atom_site.Cartn_z
   _atom_site.occupancy
   _atom_site.B_iso_or_equiv
   _atom_site.type_symbol
   _atom_site.pdbx_formal_charge
   _atom_site.label_asym_id
   _atom_site.label_entity_id
   _atom_site.label_seq_id
   _atom_site.pdbx_PDB_model_num"""


hhdb_build_template="""
cd %(msa_dir)s
ffindex_build -s ../DB_msa.ffdata ../DB_msa.ffindex .
cd %(hhDB_dir)s
ffindex_apply DB_msa.ffdata DB_msa.ffindex  -i DB_a3m.ffindex -d DB_a3m.ffdata  -- hhconsensus -M 50 -maxres 65535 -i stdin -oa3m stdout -v 0
rm DB_msa.ffdata DB_msa.ffindex
ffindex_apply DB_a3m.ffdata DB_a3m.ffindex -i DB_hhm.ffindex -d DB_hhm.ffdata -- hhmake -i stdin -o stdout -v 0
cstranslate -f -x 0.3 -c 4 -I a3m -i DB_a3m -o DB_cs219 
sort -k3 -n -r DB_cs219.ffindex | cut -f1 > sorting.dat

ffindex_order sorting.dat DB_hhm.ffdata DB_hhm.ffindex DB_hhm_ordered.ffdata DB_hhm_ordered.ffindex
mv DB_hhm_ordered.ffindex DB_hhm.ffindex
mv DB_hhm_ordered.ffdata DB_hhm.ffdata

ffindex_order sorting.dat DB_a3m.ffdata DB_a3m.ffindex DB_a3m_ordered.ffdata DB_a3m_ordered.ffindex
mv DB_a3m_ordered.ffindex DB_a3m.ffindex
mv DB_a3m_ordered.ffdata DB_a3m.ffdata
cd %(home_path)s
"""

def parse_args(expert=False):
    """setup program options parsing"""
    parser = OptionParser(usage="Usage: gapTrick [options]", version=version.__version__)


    required_opts = OptionGroup(parser, "Required parameters")
    parser.add_option_group(required_opts)


    required_opts.add_option("--seqin", action="store", \
                            dest="seqin", type="string", metavar="FILENAME", \
                  help="Fasta file with target sequences. Corresponding (unique) MSAs will be acquired from the mmseqs2 API", \
                                default=None)

    required_opts.add_option("--template", action="store", \
                            dest="templates", type="string", metavar="FILENAME", \
                  help="template in mmCIF or PDB format", default=None)

    required_opts.add_option("--jobname", action="store", dest="jobname", type="string", metavar="DIRECTORY", \
                  help="output directory name", default=None)

    extra_opts = OptionGroup(parser, "Extra parameters")
    parser.add_option_group(extra_opts)

    extra_opts.add_option("--relax", action="store_true", dest="relax", default=False, \
                  help="relax top model")

    extra_opts.add_option("--msa_dir", action="store", \
                            dest="msa_dir", type="string", metavar="DIRNAME", \
                  help="directory with precomputed MSAs for recycling. It assumes that first line in a MSA is a target sequence", \
                                default=None)
    extra_opts.add_option("--chain_ids", action="store", \
                            dest="chain_ids", type="string", metavar="CHAR,CHAR", \
                  help="comma-separated template chains corresponding to consequtive MSAs", default=None)

    extra_opts.add_option("--max_seq", action="store", dest="max_seq", type="int", metavar="INT", \
                  help="maximum number of MSA seqeunces", default=5000)

    extra_opts.add_option("--data_dir", action="store", dest="data_dir", type="string", metavar="DIRNAME", \
                  help="Path to AlphaFold2 parameters", default=None)



    #BENCHMARKS ONLY!
    expert_opts = OptionGroup(parser, "Expert parameters",
                                            "Defaults are optimal in most of "
                                            "the cases. Enable with -e/--expert")
    parser.add_option_group(expert_opts)

    expert_opts.add_option("-e", "--expert", action="store_true", dest="expert", default=None, \
                  help=SUPPRESS_HELP)

    expert_opts.add_option("--debug", action="store_true", dest="debug", default=False, \
                  help=SUPPRESS_HELP if not expert else "write more on output")

    expert_opts.add_option("--truncate", action="store", dest="truncate", type="float", metavar="FLOAT", \
                  help=SUPPRESS_HELP if not expert else "remove a fraction of truncate residues from each continuous chain fragment in a template", default=None)

    expert_opts.add_option("--iterate", action="store", dest="iterate", type="int", metavar="INT", \
                  help=SUPPRESS_HELP if not expert else "re-iterate prediction [default %default]", default=1)

    expert_opts.add_option("--pbty_cutoff", action="store", dest="pbty_cutoff", type="float", metavar="FLOAT", \
                  help=SUPPRESS_HELP if not expert else "Probability cutoff for the contact identification [default %default]", default=0.8)

    expert_opts.add_option("--plddt_cutoff", action="store", dest="plddt_cutoff", type="float", metavar="FLOAT", \
                  help=SUPPRESS_HELP if not expert else "AF templates only; removes all residues with plddt (B-factor) below given threshold [default %default]",
                  default=None)

    expert_opts.add_option("--rotrans", action="store", dest="rotrans", type="string", metavar="FLOAT,FLOAT", \
                  help=SUPPRESS_HELP if not expert else "rotate/translate template chains about their COMs up to --rotran=angle,distance", default=None)

    extra_opts.add_option("--fixed_chain_ids", action="store", \
                            dest="fixed_chain_ids", type="string", metavar="CHAR,CHAR", \
                  help=SUPPRESS_HELP if not expert else "comma-separated template chains to be exluded from rot/trans", default=None)

    expert_opts.add_option("--cardinality", action="store", dest="cardinality", type="string", metavar="INT,INT", \
                  help=SUPPRESS_HELP if not expert else "cardinalities of consecutive MSA", default=None)

    expert_opts.add_option("--trim", action="store", dest="trim", type="string", metavar="INT,INT", \
                  help=SUPPRESS_HELP if not expert else "lengths of consecutive target seqs", default=None)

    expert_opts.add_option("--preds_per_model", action="store", dest="preds_per_model", type="int", metavar="INT", \
                  help=SUPPRESS_HELP if not expert else "number of predictions per model (default: %default)", default=1)

    expert_opts.add_option("--num_recycle", action="store", dest="num_recycle", type="int", metavar="INT", \
                  help=SUPPRESS_HELP if not expert else "number of recycles", default=3)

    expert_opts.add_option("--seed", action="store", dest="seed", type="int", metavar="INT", \
                  help=SUPPRESS_HELP if not expert else "random seed (default None)", default=None)

    expert_opts.add_option("--dryrun", action="store_true", dest="dryrun", default=False, \
                  help=SUPPRESS_HELP if not expert else "check template alignments and quit")

    expert_opts.add_option("--keepalldata", action="store_true", dest="keepalldata", default=False, \
                  help=SUPPRESS_HELP if not expert else "keep all output data (MSAs, pkls, etc), you wont need them in most of the cases")

    expert_opts.add_option("--nomerge", action="store_true", dest="nomerge", default=False, \
                  help=SUPPRESS_HELP if not expert else "Use input templates as monomers. Benchmarks only!")

    expert_opts.add_option("--noseq", action="store_true", dest="noseq", default=False, \
                  help=SUPPRESS_HELP if not expert else "Mask template sequence (replace residue ids with gaps and add missing CBs)")

    expert_opts.add_option("--msa", action="store", dest="msa", type="string", metavar="FILENAME,FILENAME", \
                  help=SUPPRESS_HELP if not expert else "comma-separated a3m MSAs. First sequence is a target", default=None)

    (options, _args)  = parser.parse_args()
    return (parser, options)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def query_mmseqs2(query_sequence, msa_fname, use_env=False, filter=False, user_agent=''):

    def submit(query_sequence, mode):
        while True:
            try:
                res = requests.post(f'{MMSEQS_API_SERVER}/ticket/msa', data={'q':f">1\n{query_sequence}", 'mode': mode}, timeout=12.01, headers=headers)
            except requests.exceptions.Timeout:
                logger.info("MMSeqs2 API submission timeout. Retrying...")
                continue
            except Exception as e:
                logger.info(f"MMSeqs2 API submission error: {e}")
                time.sleep(5)
                continue
            break

        return res.json()

    def status(ID):
        while True:
            try:
                res = requests.get(f'{MMSEQS_API_SERVER}/ticket/{ID}', timeout=12.01, headers=headers)
            except requests.exceptions.Timeout:
                logger.info("MMSeqs2 API status timeout. Retrying...")
                continue
            except Exception as e:
                logger.info(f"MMSeqs2 API status error: {e}")
                time.sleep(5)
                continue
            break

        return res.json()

    def download(ID, path):
        while True:
            try:
                res = requests.get(f'{MMSEQS_API_SERVER}/result/download/{ID}', timeout=12.01, headers=headers)
            except requests.exceptions.Timeout:
                logger.info("MMSeqs2 API download timeout. Retrying...")
                continue
            except Exception as e:
                logger.info(f"MMSeqs2 API download error: {e}")
                time.sleep(5)
                continue
            break

        with open(path,"wb") as out: out.write(res.content)

    # ------------

    headers = {'User-Agent':user_agent}

    if filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    logger.info(f" --> MMSeqs2 API query:")
    pretty_sequence_print(name_a="        ", seq_a=query_sequence)
    logger.info(f"     MMSeqs2 API output file: {msa_fname}")

    if os.path.isfile(msa_fname):
        logger.info(f"Output file {msa_fname} already exists!")
        logger.info("")
        return 0

    with tempfile.TemporaryDirectory() as tmp_path:
        tar_gz_file = os.path.join(tmp_path, 'out.tar.gz')
        if not os.path.isfile(tar_gz_file):
            out = submit(query_sequence, mode)
            while out["status"] in ["UNKNOWN","RUNNING","PENDING"]:
                logger.info(f'     MMSeqs2 API status: {out["status"]}')
                time.sleep(10)
                out = status(out["id"])

            logger.info(f'     MMSeqs2 API status: {out["status"]}')

            if out["status"]=="RATELIMIT": 
                print("ERROR: MMseqs2 API request rejected (too many connections). Try again later...")
                exit(0)

            download(out["id"], tar_gz_file)

        # parse a3m files
        with tarfile.open(tar_gz_file) as tar_gz: tar_gz.extractall(tmp_path)

        a3m_files = [os.path.join(tmp_path, "uniref.a3m")]
        if use_env: a3m_files.append( os.path.join(tmp_path, "bfd.mgnify30.metaeuk30.smag30.a3m") )

        with open(msa_fname,"w") as a3m_out:
            for a3m_file in a3m_files:
                for line in open(a3m_file,"r"):
                    line = line.replace("\x00","")
                    if len(line) > 0:
                        a3m_out.write(line)

    logger.info(f"     Successfully created {msa_fname}")
    logger.info("")


    return 0

# -----------------------------------------------------------------------------

def save_pdb(structure, ofname):
    pdbio = MMCIFIO()
    pdbio.set_structure(structure)
    with Path(ofname).open('w') as of:
        pdbio.save(of)

# -----------------------------------------------------------------------------

def CB_xyz(n, ca, c):
    bondl=1.52
    rada=1.93
    radd=-2.14

    vec_nca = (n-ca)/np.linalg.norm(n-ca)
    vec_cca = (c-ca)/np.linalg.norm(c-ca)

    normal_vec = np.cross(vec_nca, vec_cca)

    m = [vec_nca, np.cross(normal_vec, vec_nca), normal_vec]
    d = [np.cos(rada), np.sin(rada)*np.cos(radd), -np.sin(rada)*np.sin(radd)]
    return c + sum([bondl*_m*_d for _m,_d in zip(m,d)])


# -----------------------------------------------------------------------------

def get_CAs(structure, sel_residx=None):
    '''
        these multi-chain objects should preserve residue order
            from merged 1-chain prediction (and a template)
    '''
    CA_atoms = []
    residx=0
    for _chain in structure :
        for _res in _chain:
            if sel_residx and residx in sel_residx:
                CA_atoms.append(_res['CA'])
            residx+=1
    return CA_atoms

# -----------------------------------------------------------------------------

def predict_structure(prefix,
                      query_sequence,
                      feature_dict,
                      Ls,
                      model_params,
                      model_runner_1,
                      model_runner_3,
                      do_relax                  =   False,
                      model2template_mappings   =   None,
                      random_seed               =   None,
                      gap_size                  =   200,
                      template_fn_list          =   []):

    if random_seed is None:
        random_seed = np.random.randint(sys.maxsize//5)

    logger.info("")
    logger.info(f"Random seed: {random_seed}")
    logger.info("")

    inputpath=Path(prefix, "input")
    outputpath=Path(prefix, "output")

    seq_len = len(query_sequence)

    idx_res = feature_dict['residue_index']
    L_prev = 0

    # Ls: number of residues in each chain
    for L_i in Ls[:-1]:
        idx_res[L_prev+L_i:] += gap_size
        L_prev += L_i
    chains = list("".join([string.ascii_uppercase[n]*L for n,L in enumerate(Ls)]))
    feature_dict['residue_index'] = idx_res

    plddts,ptmscore= [],[]
    unrelaxed_pdb_lines = []
    model_names = []

    for imodel, (model_name, params) in enumerate(model_params.items()):
        logger.info(f" --> Running {model_name} ({imodel+1} of {len(model_params)})")

        if re.search(r'model_[12]', model_name):
            model_runner = model_runner_1
        else:
            model_runner = model_runner_3

        model_runner.params = params

        input_features = model_runner.process_features(feature_dict, random_seed=random_seed+imodel)

        prediction_result = model_runner.predict(input_features, random_seed=random_seed+imodel)
        mean_plddt = np.mean(prediction_result["plddt"][:seq_len])
        mean_ptm = prediction_result["ptm"]

        final_atom_mask = prediction_result["structure_module"]["final_atom_mask"]
        b_factors = prediction_result["plddt"][:, None] * final_atom_mask

        resid2chain = {}
        input_features["asym_id"] = feature_dict["asym_id"] - feature_dict["asym_id"][...,0]
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
                                            remove_leading_feature_dimension=False)

        unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
        plddts.append(prediction_result["plddt"][:seq_len])
        ptmscore.append(prediction_result["ptm"])
        model_names.append(model_name)

        with Path(outputpath, f'unrelaxed_{model_name}.pdb').open('w') as of: of.write(unrelaxed_pdb_lines[-1])

        logger.info(f"     <pLDDT>={np.mean(prediction_result['plddt'][:seq_len]):6.4f} pTM={prediction_result['ptm']:6.4f}")

        outdict={'predicted_aligned_error' : prediction_result['predicted_aligned_error'], \
                 'ptm'                     : prediction_result['ptm'], \
                 'plddt'                   : prediction_result['plddt'][:seq_len], \
                 'distogram'               : prediction_result['distogram']}

        with Path(outputpath, f'unrelaxed_{model_name}_pae.json').open('w') as of:
            of.write(json.dumps([{'predicted_aligned_error':prediction_result['predicted_aligned_error'].astype(int).tolist()}]))

        with Path(outputpath, f"result_{model_name}.pkl").open('wb') as of:
            pickle.dump(outdict, of, protocol=pickle.HIGHEST_PROTOCOL)

    # rerank models based on pTM (not predicted lddt!)
    ptm_rank = np.argsort(ptmscore)[::-1]


    ranking_debug_dict = {'ptm':{}, 'order':[]}

    logger.info("")
    logger.info(f"Reranking models based on pTM score: {ptm_rank}")
    for n,_idx in enumerate(ptm_rank):

        # relax TOP model only
        if do_relax and n<1:

            pdb_obj = protein.from_pdb_string(unrelaxed_pdb_lines[_idx])

            logger.info(f"Starting Amber relaxation for {model_names[_idx]}")
            start_time = time.time()

            amber_relaxer = relax.AmberRelaxation(
                                    max_iterations=0,
                                    tolerance=2.39,
                                    stiffness=10.0,
                                    exclude_residues=[],
                                    max_outer_iterations=3,
                                    use_gpu=True)

            _pdb_lines, _, _ = amber_relaxer.process(prot=pdb_obj)

            logger.info(f"Done, relaxation took {(time.time() - start_time):.1f}s")


        else:
            _pdb_lines = unrelaxed_pdb_lines[_idx]


        ranking_debug_dict['order'].append(model_names[_idx])
        ranking_debug_dict['ptm'][model_names[_idx]]=float(ptmscore[_idx])

        pdb_fn = f"ranked_{n}.pdb"
        Path(outputpath, f'unrelaxed_{model_names[_idx]}_pae.json').rename( Path(outputpath, f'ranked_{n}_pae.json') )

        logger.info(f"{pdb_fn} ({model_names[_idx]}) <pLDDT>={np.mean(plddts[_idx]):6.4f} pTM={ptmscore[_idx]:6.4f}")

        pdb_header     = [ f"REMARK 0" ]
        pdb_header.append( f"REMARK 0 MODEL PREDICTED WITH ALPHAFOLD2/gapTRICK ON {datetime.now().strftime('%H:%M %d/%m/%Y')}")

        pdb_header.append( f"REMARK 0" )
        pdb_header.append( f"REMARK 0 TEMPLATES {','.join(template_fn_list)}" )
        pdb_header.append( f"REMARK 0 MEAN pLDDT {np.mean(plddts[_idx]):6.4f}" )
        pdb_header.append( f"REMARK 0 pTM {ptmscore[_idx]:6.4f}" )
        if do_relax and n<1: pdb_header.append("REMARK 0 ENERGY MINIMIZED WITH AMBER")
        pdb_header.append( f"REMARK 0" )

        pdb_header = "\n".join(pdb_header)

        #superpose final models onto a template (first, if there is more of them...)
        if model2template_mappings:
            tpl_fn,residx_mappings = list(model2template_mappings.items())[0]
            template_structure = parse_pdb_bio(Path(inputpath, f"{tpl_fn}.cif"), outid=tpl_fn)

            with io.StringIO(_pdb_lines) as outstr:
                parser = PDBParser(QUIET=True)
                model_structure = parser.get_structure(id='xyz', file=outstr)[0]

            template_CAs = get_CAs(template_structure, list(residx_mappings.values()))
            model_CAs = get_CAs(model_structure, list(residx_mappings.keys()))

            sup = Superimposer()
            sup.set_atoms(template_CAs, model_CAs)
            sup.apply(model_structure)

            pdbio = PDBIO()
            pdbio.set_structure(model_structure)
            with Path(outputpath, pdb_fn).open('w') as of:
                of.write(f"{pdb_header}\n")
                pdbio.save(of)

        else:
            with Path(outputpath, pdb_fn).open('w') as of:
                of.write(f"{pdb_header}\n")
                of.write(_pdb_lines)

    # save a file with pTMs and rankings
    with Path(outputpath, 'ranking_debug.json').open('w') as of:
        json.dump(ranking_debug_dict, of)

# -----------------------------------------------------------------------------                    

def make_figures(prefix, print_contacts=False, keepalldata=False, pbty_cutoff=0.8, distance_cutoff=8.0):

    datadir=Path(prefix, "output")
    figures_dir = Path(prefix, "figures")
    figures_dir.mkdir(parents=True, exist_ok=False)

    af2o = plotter()
    datadict = af2o.parse_model_pickles(datadir, verbose=False)

    # PAE
    ff=af2o.plot_predicted_alignment_error(datadict)
    ff.savefig(fname=os.path.join(figures_dir, f"pae.png"), dpi=150, bbox_inches = 'tight')
    ff.savefig(fname=os.path.join(figures_dir, f"pae.svg"), bbox_inches = 'tight')

    # pLDDT
    ff=af2o.plot_plddts(datadict)
    ff.savefig(fname=os.path.join(figures_dir, f"plddt.png"), dpi=150, bbox_inches = 'tight')
    ff.savefig(fname=os.path.join(figures_dir, f"plddt.svg"), bbox_inches = 'tight')

    # distogram
    ff,contacts_txt = af2o.plot_distogram(datadict, distance=distance_cutoff, print_contacts=False, pbtycutoff=pbty_cutoff)
    ff.savefig(fname=os.path.join(figures_dir, f"distogram.png"), dpi=150, bbox_inches = 'tight')
    ff.savefig(fname=os.path.join(figures_dir, f"distogram.svg"), bbox_inches = 'tight')

    msa_dir = Path(prefix, 'msa')

    if msa_dir.exists():
        a3m_filenames = glob.glob( os.path.join(msa_dir, '*.a3m') )
        if a3m_filenames:
            ff = af2o.msa2fig(a3m_filenames=a3m_filenames)
            ff.savefig(fname=os.path.join(figures_dir, f"msa.png"), dpi=150, bbox_inches = 'tight')
            ff.savefig(fname=os.path.join(figures_dir, f"msa.svg"), bbox_inches = 'tight')

# -----------------------------------------------------------------------------                    
# lists likely contacts and generates pymol/chimera scripts
# bypasses af2plots and has no matplolib dep

def make_contact_scripts(prefix, feature_dict, print_contacts=False, keepalldata=False, pbty_cutoff=0.8, distance_cutoff=8.0):

    datadir=Path(prefix, "output")
    datadict = {}

    for fn in glob.glob("%s/result*.pkl" % datadir):
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

    distance_bins = [(0, bin_edges[0])]
    distance_bins += [(bin_edges[idx], bin_edges[idx + 1]) for idx in range(len(bin_edges) - 1)]
    distance_bins.append((bin_edges[-1], np.inf))
    distance_bins = tuple(distance_bins)
    print()
    print(f"AlphaFold2 distogram distance range [{bin_edges[0]}, {bin_edges[-1]}]")
    print()
    # truncate distance to the available range
    distance = np.clip(distance_cutoff, 3, 20)

    bin_idx=np.max(np.where(bin_edges<distance))


    below8pbty = np.sum(probs, axis=2, where=(np.arange(probs.shape[-1])<bin_idx))

    requested_contacts=[]
    if print_contacts:
        print()
        print(f"AlphaFold2-predicted contacts below {distance}A with estimated probability (*-inter chains)")

    chain_ids = string.ascii_uppercase
    chain_lens = []
    for i in range(assembly_num_chains):
        chain_lens.append(np.sum(np.array(asym_id)==(i+1)))

    chain_lens = np.array(chain_lens)
    resi_i,resi_j = np.where(below8pbty>pbty_cutoff)
    for i,j in zip(resi_i, resi_j):

        ci = int(asym_id[i]-1)
        cj = int(asym_id[j]-1)

        # skipp: close, diag, and symm
        if i==j: continue
        if np.abs(i-j)<2 and ci==cj: continue
        if ci>cj: continue

        reli = 1+i-sum(chain_lens[:ci])
        relj = 1+j-sum(chain_lens[:cj])

        requested_contacts.append(f"{reli}/{chain_ids[ci]} {relj}/{chain_ids[cj]} {below8pbty[i,j]}")

        if print_contacts: print(f"{'*' if ci!=cj else ' '} {reli:-4d}/{chain_ids[ci]} {relj:-4d}/{chain_ids[cj]} {below8pbty[i,j]:5.2f}")

    # contacts list
    contact_template = r"^(?P<res1>\w+?)/(?P<ch1>\w+?)\s+(?P<res2>\w+?)/(?P<ch2>\w+?)\s+(?P<pbty>[\d\.]*?)$"
    structure = parse_pdb_bio(Path(prefix, "output", "ranked_0.pdb"), outid="XYZ", remove_alt_confs=True)
    protein = get_prot_chains_bio(structure)
    chain_seq_dict = {}
    for chain in protein:
        chain_seq_dict[chain.id]="".join([ogt[_r.get_resname()] for _r in chain.get_unpacked_list()])

    idx=0
    d={}
    d['modelid']="ranked_0"
    d['A_atom_name']='CA'
    d['B_atom_name']='CA'

    if keepalldata: pymol_all = [pymol_header%d]
    pymol_int = [pymol_header%d]
    chimerax_int = []
    pymol_sb_int = [pymol_header%d]
    chimerax_sb_int = []

    contacts_list = []
    interchain_contacts_list = []
    interchain_sb_list = []

    for contact_str in requested_contacts:
        m = re.match(contact_template, contact_str)
        d['A_chain'] = ci = m.group('ch1')
        d['B_chain'] = cj = m.group('ch2')
        d['A_resid'] = resi = m.group('res1')
        d['B_resid'] = resj = m.group('res2')

        resni = tgo[chain_seq_dict[ci][int(resi)-1]].upper()
        resnj = tgo[chain_seq_dict[cj][int(resj)-1]].upper()

        if resni=='GLY':
            d['A_atom_name']='CA'
        else:
            d['A_atom_name']='CB'

        if resnj=='GLY':
            d['B_atom_name']='CA'
        else:
            d['B_atom_name']='CB'


        _cstr = f"""{'*' if ci!=cj else ' '} {resni}/{ci}/{resi:4s} {resnj}/{cj}/{resj:4s} {float(m.group('pbty')):.2f}"""

        if print_contacts: logger.info(_cstr)
        contacts_list.append(_cstr)

        if ci!=cj:
            pymol_int.append("show sticks, \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s\ncolor atomic, \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s"%d)
            pymol_int.append("show sticks, \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s\ncolor atomic, \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s"%d)
            pymol_int.append(pymol_dist_generic%d)

            chimerax_int.append(chimerax_dist_generic%d)
            interchain_contacts_list.append(_cstr[2:])

            if (resnj in ['ASP', 'GLU'] and resni in ['LYS', 'ARG']) or (resni in ['ASP', 'GLU'] and resnj in ['LYS', 'ARG']):
                interchain_sb_list.append(_cstr[2:])
                pymol_sb_int.append("show sticks, \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s\ncolor atomic, \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s"%d)
                pymol_sb_int.append("show sticks, \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s\ncolor atomic, \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s"%d)
                pymol_sb_int.append(pymol_dist_generic%d)
                chimerax_sb_int.append(chimerax_dist_generic%d)

        if keepalldata:
            pymol_all.append("show sticks, \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s\ncolor atomic, \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s"%d)
            pymol_all.append("show sticks, \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s\ncolor atomic, \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s"%d)
            pymol_all.append(pymol_dist_generic%d)

        idx+=1

    if keepalldata:
        with open(os.path.join(datadir, f"pymol_all_contacts.pml"), 'w') as ofile:
            ofile.write("\n".join(pymol_all))

    with open(os.path.join(datadir, f"pymol_interchain_contacts.pml"), 'w') as ofile:
        ofile.write("\n".join(pymol_int))

    with open(os.path.join(datadir, f"chimerax_interchain_contacts.cxc"), 'w') as ofile:
        chimerax_int.append(chimerax_footer)
        ofile.write("\n".join(chimerax_int))

    if interchain_sb_list:
        with open(os.path.join(datadir, f"pymol_interchain_saltbridges.pml"), 'w') as ofile:
            ofile.write("\n".join(pymol_sb_int))

        with open(os.path.join(datadir, f"chimerax_interchain_saltbridges.cxc"), 'w') as ofile:
            chimerax_sb_int.append(chimerax_footer)
            ofile.write("\n".join(chimerax_sb_int))

    with open(os.path.join(datadir, f"contacts.txt"), 'w') as ofile:
        ofile.write("\n".join(contacts_list))

    logger.info("\n\n")

    if not interchain_contacts_list:
        logger.info(f""" ==> Found NO inter-chain contacts (dist<8A and pbty>0.8)\n"""+\
                     """     The prediction may be NOT correct\n""")
    else:
        logger.info(f""" ==> Found {len(interchain_contacts_list)} inter-chain contacts (dist<8A and pbty>0.8)\n""")

        for idx,_c in enumerate(interchain_contacts_list):
            logger.info(f"     {idx+1:03d} {_c}")
            if idx>8:
                logger.info("    [..] full list in contacts.txt")
                break
        if interchain_sb_list:
            logger.info("")
            logger.info(f"""     Among these {len(interchain_sb_list)} may form salt-bridges""")
            for idx,_c in enumerate(interchain_sb_list):
                logger.info(f"     {idx+1:03d} {_c}")
                if idx>8:
                    logger.info("    [..] full list in contacts.txt")
                    break
        else:
            logger.info("")
            logger.info(f"""     No potential salt-bridges found""")
# -----------------------------------------------------------------------------                    
                    
def match_template_chains_to_target(ph, target_sequences):
    logger.info(f" --> Greedy matching of template chains to target sequences")

    chain_dict = {}
    for chain in ph.chains():
        is_protein=False
        for conf in chain.conformers():
            if conf.is_protein(min_content=0.5):
                is_protein=True
                break
        if not is_protein: continue


        chain_dict[chain.id]="".join(chain.as_sequence())


    greedy_selection = []
    for _idx, _target_seq in enumerate(target_sequences):
        _tmp_si={}
        for cid in chain_dict:
            if cid in greedy_selection: continue
            aligner = Align.PairwiseAligner()
            alignments = aligner.align(chain_dict[cid], _target_seq)
            si = alignments[0].score
            _tmp_si[cid]=100.0*si/len(chain_seq_dict[cid])

        if _tmp_si:
            greedy_selection.append( sorted(_tmp_si.items(), key=lambda x: x[1])[-1][0] )
            logger.info(f"     #{_idx}: chain {greedy_selection[-1]} with SI={_tmp_si[greedy_selection[-1]]:.1f}",\
                           "[", ",".join([f"{k}:{v:.1f}" for k,v in _tmp_si.items()]), "]")

    if not len(greedy_selection) == len(target_sequences):
        logger.info("WARNING: template-target sequence match is incomplete!")

    logger.info("")

    return(greedy_selection)

# -----------------------------------------------------------------------------

def parse_pdb_bio(ifn, outid="xyz", plddt_cutoff=None, remove_alt_confs=False):

    class NotAlt(Select):
        def accept_atom(self, atom):
            if plddt_cutoff: 
                return (not atom.is_disordered() or atom.get_altloc() == "A") and atom.bfactor > plddt_cutoff
            else:
                return not atom.is_disordered() or atom.get_altloc() == "A"

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(outid, ifn)[0]

    except:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(outid, ifn)[0]

    if remove_alt_confs:
        with io.StringIO() as outstr:
            pdbio = MMCIFIO()
            pdbio.set_structure(structure)
            pdbio.save(outstr, select=NotAlt())
            outstr.seek(0)

            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(outid, outstr)[0]
            for chain in structure:
                for resi in chain:
                    for atom in resi:
                        atom.set_altloc(" ")

    return structure

# -----------------------------------------------------------------------------

def match_template_chains_to_target_bio(structure, target_sequences):
    logger.info(f" --> Greedy matching template chains to target sequences")

    chain_seq_dict = {}
    chain_ends_dict = {}
    protein = get_prot_chains_bio(structure)
    for chain in protein:
        chain_seq_dict[chain.id]="".join([ogt[_r.get_resname()] for _r in chain.get_unpacked_list()])
        _resis = list(chain.get_residues())
        chain_ends_dict[chain.id]= (np.array(_resis[0]['CA']), np.array(_resis[-1]['CA']))

    greedy_selection = []
    for _idx, _target_seq in enumerate(target_sequences):
        _tmp_si={}
        for cid in chain_seq_dict:
            if cid in greedy_selection: continue
            aligner = Align.PairwiseAligner()
            alignments = aligner.align(chain_seq_dict[cid], _target_seq)
            si = alignments[0].score
            _tmp_si[cid]=si#100.0*si#/min(len(chain_seq_dict[cid]),len(_target_seq))

        if _tmp_si:
            greedy_selection.append( sorted(_tmp_si.items(), key=lambda x: x[1])[-1][0] )
            other_si = "".join(["[", ",".join([f"{k}:{v:.1f}" for k,v in _tmp_si.items()]), "]"])
            logger.info(f"     #{_idx}: {greedy_selection[-1]} with SI={_tmp_si[greedy_selection[-1]]:.1f} {other_si}")

    if not len(greedy_selection) == len(target_sequences):
        logger.info("WARNING: template-target sequence match is incomplete!")

    #for c1, c2 in zip(greedy_selection[:-1], greedy_selection[1:]):
    #    print(c1, c2, np.linalg.norm(chain_ends_dict[c2][0]-chain_ends_dict[c1][1]))

    logger.info("")

    return(greedy_selection)


# -----------------------------------------------------------------------------

def get_resi_chunks(chain):
    """
        find residue ranges of continous perotein chunks in a chain
        (ignores 1-resi gaps due to SeMet)
    """

    resi_chunks = []

    resids=[_r.id[1] for _r in chain]
    for k, g in groupby(enumerate(set(resids)), lambda idx : idx[0] - idx[1]):
        chunk =list(map(itemgetter(1), g))
        if not resi_chunks:
            resi_chunks.append( [chunk[0], chunk[-1]] )
        else:
            # ignore single-resi gaps - removed SeMet
            if chunk[0]-resi_chunks[-1][-1]==2:
                resi_chunks[-1] = (resi_chunks[-1][0], chunk[-1])
            else:
                resi_chunks.append( [chunk[0], chunk[-1]] )

    return resi_chunks


def select_resi2keep(chunks, truncate=0.3):
    """
        generates list of residues to keep after removing a fraction truncate from each chain
    """

    _chunk2keep = []

    for _frag in chunks:
        chunk2cut = int(truncate*(_frag[-1]-_frag[0]))
        if np.random.uniform(0,1)>0.5:
            _chunk2keep.extend(range(_frag[0], _frag[-1]-chunk2cut))
        else:
            _chunk2keep.extend(range(_frag[0]+chunk2cut, _frag[-1]))

    return _chunk2keep 


def random_point_on_sphere():
    z = np.random.uniform(-1,1)
    t = 2.0*np.pi * np.random.uniform(0,1);
    r = np.sqrt(1.0-z*z);
    return np.array([r * np.cos(t), r * np.sin(t), z])


def get_prot_chains_bio(structure, min_prot_content=0.1, truncate=None, rotmax=None, transmax=None, fixed_chain_ids=None):
    '''
        removes non-protein chains and residues wouth CA atoms (required for superposition)
    '''
    for chain in list(structure):
        chain_len_before = len(chain)
        for res in list(chain):
            # a residue must be an amino-acid and contain CA atom
            if not (res.get_resname() in ogt.keys() and 'CA' in [_.name.strip() for _ in res]):
                chain.detach_child(res.id)
        if (chain_len_before-len(chain))/chain_len_before>(1.0-min_prot_content):
            logger.info(f'WARNING: removed non-protein template chain {chain.id}')
            chain.parent.detach_child(chain.id)

    assert len(structure), f"Template structure must contain at least one protein chain (>{100*min_prot_content:.1f}% amino acid residues)"

    if truncate:
        logger.info(f"\nWARNING: Removed {100*truncate:.0f}% residues from template!\n")
        resi2keep = {}
        for chain in structure:
            _ch = get_resi_chunks(chain)
            _a = resi2keep.setdefault(chain.id, [])
            _a.extend( select_resi2keep(_ch, truncate=truncate) )

        for chain in list(structure):
            chain_len_before = len(chain)
            for res in list(chain):
                if not res.id[1] in resi2keep[chain.id]:
                    chain.detach_child(res.id)


    if rotmax and transmax:
        logger.info("")
        for chain in structure:

            if fixed_chain_ids and chain.id in fixed_chain_ids.split(","): continue

            com_vec = Vector(np.array([atom.get_coord() for atom in chain.get_atoms()]).mean(axis=0))
            axis = random_point_on_sphere()
            angle = np.random.uniform(0,1) * ( np.pi - 0.001 ) * rotmax/180
            trans = Vector(np.array(random_point_on_sphere())*np.random.uniform(0,1)*transmax)
            rot = rotaxis2m(angle, Vector(axis))
            logger.info(f"WARNING: Chain {chain.id} rotated/translated by {180*angle/np.pi:4.2f} deg and {trans.norm():4.2f} A")
            for atom in chain.get_atoms():
                atom.set_coord( (Vector(atom.coord)-com_vec).left_multiply(rot) + trans + com_vec )
        logger.info("")

    return structure

# -----------------------------------------------------------------------------

def chain2CIF_bio(chain, outid, outfn):

    poly_seq_block = []

    seq = "".join( [ogt[_r.get_resname()] for _r in chain] )
    poly_seq_block.append("#")
    poly_seq_block.append("loop_")
    poly_seq_block.append("_entity_poly_seq.entity_id")
    poly_seq_block.append("_entity_poly_seq.num")
    poly_seq_block.append("_entity_poly_seq.mon_id")
    poly_seq_block.append("_entity_poly_seq.hetero")
    for i, aa in enumerate(seq):
        three_letter_aa = tgo[aa]
        poly_seq_block.append(f"0\t{i + 1}\t{three_letter_aa}\tn")

    with open(outfn, 'w') as of:
        # sequence
        print(FAKE_MMCIF_HEADER%locals(), file=of)
        print("\n".join(poly_seq_block), file=of)

        # atom block header
        print(MMCIF_ATOM_BLOCK_HEADER, file=of)

        # and atom details
        atom_idx=1
        for res_idx,res in enumerate(chain):
            for atom in res:
                print(f"   ATOM   {atom_idx:5} {atom.name:5} . {res.resname:4} {chain.id:3} {res._id[1]:5}"+\
                        f" ? {atom.coord[0]:10.5f} {atom.coord[1]:10.5f} {atom.coord[2]:10.5f} {atom.occupancy:6.3f}"+\
                      f" {atom.bfactor:9.5f}  {atom.element:3} ? {chain.id:2} ? {res_idx+1:5} 1", file=of)
                atom_idx+=1

# -----------------------------------------------------------------------------

def template_preps_bio(template_fn_list,
                       chain_ids,
                       target_sequences,
                       outpath          =   None,
                       resi_shift       =   200,
                       truncate         =   None,
                       plddt_cutoff     =   None,
                       rotmax           =   None,
                       transmax         =   None,
                       fixed_chain_ids  =   None):
    '''
        BioPython version: this will generate a merged, single-chain template in a AF2-compatible mmCIF file(s)
    '''

    converted_template_fns = []
    template2input_mapping = {}

    idx=0
    for ifn in template_fn_list:
        outid=f"{idx:04d}"
        _ph = parse_pdb_bio(ifn, outid=outid, plddt_cutoff=plddt_cutoff, remove_alt_confs=True)

        # save input template
        save_pdb(_ph, os.path.join(outpath, f"{outid}_inp.cif"))

        # extarct protein chains and bias the template (if requested)
        prot_ph = get_prot_chains_bio(_ph, truncate=truncate, rotmax=rotmax, transmax=transmax, fixed_chain_ids=fixed_chain_ids)

        # save modified template (before merging chains)
        save_pdb(prot_ph, os.path.join(outpath, f"{outid}_mod.cif"))

        if chain_ids is None:
            selected_chids = match_template_chains_to_target_bio(prot_ph, target_sequences)
        else:
            selected_chids = chain_ids.split(',')

        chaindict={}
        for ch in prot_ph:
            chaindict[ch.id]=ch
        # assembly with BioPython
        tmp_io = None
        _template2input = {}
        for ich,chid in enumerate(selected_chids):
            chain = chaindict[chid]
            chain.detach_parent()
            chain.id = "A"

            if ich==0:
                last_resid=1
            else:
                last_resid = tmp_io.get_unpacked_list()[-1]._id[1]+resi_shift

            for residx,res in enumerate(chain):
                _id = res._id
                res._id = (_id[0], last_resid+residx, _id[1])
                _template2input[last_resid+residx] = (chid, _id[1])

            if ich==0:
                tmp_io = chain
            else:
                for resi in chain:
                    tmp_io.add(resi)

        template2input_mapping[ifn]=_template2input

        if not outpath: continue

        converted_template_fns.append(os.path.join(outpath, f"{outid}.cif"))
        chain2CIF_bio(tmp_io, outid, converted_template_fns[-1])

        idx+=1

    return converted_template_fns, template2input_mapping



# -----------------------------------------------------------------------------

def template_preps_nomerge_bio(template_fn_list, chain_ids, target_sequences, outpath=None, truncate=None, plddt_cutoff=None, rotmax=None, transmax=None):
    '''
        this one will put each requested chain from each template in a separate AF2-compatible mmCIF
    '''
    converted_template_fns=[]

    idx=0
    for ifn in template_fn_list:
        _ph = parse_pdb_bio(ifn, plddt_cutoff=plddt_cutoff, remove_alt_confs=True)
        prot_ph = get_prot_chains_bio(_ph, truncate=truncate, rotmax=rotmax, transmax=transmax)

        if chain_ids is None:
            selected_chids = match_template_chains_to_target_bio(prot_ph, target_sequences)
        else:
            selected_chids = chain_ids.split(',')

        chaindict={}
        for ch in prot_ph:
            chaindict[ch.id]=ch

        # assembly with BioPython
        for chid in selected_chids:
            chain = chaindict[chid]
            chain.detach_parent()
            chain.id = "A"

            outid=f"{idx:04d}"
            if not outpath: continue

            converted_template_fns.append(os.path.join(outpath, f"{outid}.cif"))
            chain2CIF_bio(chain, outid, converted_template_fns[-1])

            idx+=1

    return converted_template_fns

# -----------------------------------------------------------------------------

def pretty_sequence_print(name_a, seq_a, name_b=None, seq_b=None, block_width=80):

    #if seq_b: assert len(seq_a) == len(seq_b)

    length = len(seq_a)
    n_blocks = length//block_width

    for ii in range(n_blocks+1):
        logger.info(f"{name_a} {seq_a[ii*block_width:(ii+1)*block_width]}")
        if seq_b:
            logger.info(f"{name_b} {seq_b[ii*block_width:(ii+1)*block_width]}")
            logger.info("")

# -----------------------------------------------------------------------------

def generate_template_features(query_sequence, db_path, template_fn_list, nomerge=False, dryrun=False, noseq=False, debug=False):
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
        logger.info(f"Template file: {template_fn}")
        filepath=Path(template_fn)
        with open(filepath, "r") as fh:
            filestr = fh.read()
            mmcif_obj = mmcif_parsing.parse(file_id=filepath.stem,mmcif_string=filestr, catch_all_errors=True)
            mmcif = mmcif_obj.mmcif_object

        if not mmcif: logger.info(mmcif_obj)

        # broken in AlphaFold/2.3.2-foss-2023a-CUDA-12.1.1
        #for chain_id,template_sequence in mmcif.chain_to_seqres.items():
        for chain in mmcif.structure:
            chain_id = chain.id
            template_sequence = "".join([ogt[_r.resname] for _r in chain.get_residues()])
            pretty_sequence_print(name_a=f"{chain_id:8s}", seq_a=template_sequence)

            seq_name = filepath.stem.upper()+"_"+chain_id
            seq = SeqRecord(Seq(template_sequence),id=seq_name,name="",description="")

            with  Path(fasta_dir,seq.id+".fasta").open("w") as fh:
                SeqIO.write([seq], fh, "fasta")

        template_seq=seq
        template_seq_path = Path(msa_dir,"template.fasta")
        with template_seq_path.open("w") as fh:
            SeqIO.write([seq], fh, "fasta")

        cmd=hhdb_build_template%locals()

        ppipe = subprocess.Popen( cmd,
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)

        for stdout_line in iter(ppipe.stdout.readline, ""):
            if debug: logger.info(stdout_line.strip())

        retcode = subprocess.Popen.wait(ppipe)


        hhsearch_runner = hhsearch.HHSearch(binary_path="hhsearch", databases=[hhDB_dir.as_posix()+"/"+db_prefix])
        with io.StringIO() as fh:
            SeqIO.write([query_seq], fh, "fasta")
            seq_fasta = fh.getvalue()

        hhsearch_result = hhsearch_runner.query(seq_fasta)
        hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)

        if len(hhsearch_hits) >0:
            logger.info("")
            logger.info(" --> Aligning template to the target sequence")
            naligned=[]
            for _i,_h in enumerate(hhsearch_hits):
                naligned.append(len(_h.hit_sequence)-_h.hit_sequence.count('-'))
                logger.info(f"     #{_i+1} aligned {naligned[-1]} out of {len(query_sequence)} residues [sum_probs={_h.sum_probs}]")
                if debug: pretty_sequence_print(name_a="target  ",
                            seq_a=query_sequence[:_h.indices_query[0]]+_h.query+query_sequence[_h.indices_query[-1]+1:],
                            name_b="template",
                            seq_b=f"{'-'*_h.indices_query[0]}{_h.hit_sequence}{'-'*(len(query_sequence)-_h.indices_query[-1]-1)}")
            logger.info("")

            # in no-merge mode accept multiple alignments, in case target is a homomultimer
            if nomerge:
                for _i,_h in enumerate(hhsearch_hits):
                    if naligned[_i]/len(template_sequence)<0.5: continue
                    logger.info(f' --> Selected alignment #{_i+1}')
                    template_hit_list.append([mmcif,_h])
            else:
                logger.info(f' --> Selected alignment #{np.argmax(naligned)+1}')
                hit = hhsearch_hits[np.argmax(naligned)]
                hit = replace(hit,**{"name":template_seq.id})

                template_hit_list.append([mmcif, hit])
        logger.info("")

    logger.info("")
    if dryrun: exit(1)

    template_features = {}
    for template_feature_name in TEMPLATE_FEATURES:
      template_features[template_feature_name] = []

    model2template_mappings={}

    for mmcif,hit in template_hit_list:

        hit_pdb_code, hit_chain_id = _get_pdb_id_and_chain(hit)
        mapping = _build_query_to_hit_index_mapping(hit.query, hit.hit_sequence, hit.indices_hit, hit.indices_query,query_sequence)

        model2template_mappings[mmcif.file_id] = dict([(q,t) for q,t in zip(hit.indices_query, hit.indices_hit) if q>0 and t>0])

        logger.info(f">{hit.name}")
        pretty_sequence_print(name_a="target  ", seq_a=query_sequence[:hit.indices_query[0]]+hit.query+query_sequence[hit.indices_query[-1]+1:],
            name_b="template", seq_b=f"{'-'*hit.indices_query[0]}{hit.hit_sequence}{'-'*(len(query_sequence)-hit.indices_query[-1]-1)}")

        # handles nomerge+noseq and other weird cases
        template_idxs = hit.indices_hit
        query_idxs = hit.indices_query

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

        if noseq: # remove sequence-related features
            logger.info("")
            logger.info("WARNING: sequence information in a template has been masked")
            logger.info("")

            features['template_sum_probs'] = [0]

            # generate a gap-only sequence
            _seq='-'*len(query_seq)

            # crate protein object from biopython strurture
            with io.StringIO() as outstr:
                _io=PDBIO()
                _io.set_structure(mmcif.structure)
                _io.save(outstr)
                outstr.seek(0)
                template_prot = protein.from_pdb_string(outstr.read())

            # mask side-chains
            masked_coords = np.zeros([1,len(query_seq), 37, 3])
            masked_coords[0, query_idxs, :5] = template_prot.atom_positions[template_idxs,:5]

            # add missing CBs
            bb_idxs = [q for t,q in zip(template_idxs,query_idxs) if jnp.all(template_prot.atom_mask[t,[0,1,2]] == 1)]
            backbone_modelled = np.full(len(query_seq), False)
            backbone_modelled[bb_idxs] = True

            missing_cb = [i for (i,b,m) in zip(bb_idxs, backbone_modelled, template_prot.atom_mask) if m[3] == 0 and b]
            missing_cb = [q for (t,q) in zip(template_idxs,query_idxs) if template_prot.atom_mask[t][3] == 0 and backbone_modelled[q] ]
            cbs = np.array([CB_xyz(masked_coords[0,_,0], masked_coords[0,_,1], masked_coords[0,_,2]) for _ in missing_cb])
            masked_coords[0, missing_cb, 3] = cbs

            atom_mask = np.zeros([1, len(query_seq), 37])
            atom_mask[0, query_idxs, :5] = template_prot.atom_mask[template_idxs,:5]

            features["template_aatype"]             =   \
                    residue_constants.sequence_to_onehot(_seq, residue_constants.HHBLITS_AA_TO_ID)[None],
            features["template_all_atom_masks"]     =   atom_mask
            features["template_all_atom_positions"] =   masked_coords
            features["template_domain_names"]       =   np.asarray(["None"])

        single_hit_result = SingleHitResult(features=features, error=None, warning=None)

        for k in template_features:
            if isinstance(template_features[k], (np.ndarray, np.generic) ):
                template_features[k] = np.append(template_features[k], features[k])
            else:
                template_features[k].append(features[k])

        for name in template_features:
            template_features[name] = np.stack(template_features[name], axis=0).astype(TEMPLATE_FEATURES[name])

    for key,value in template_features.items():
        if np.all(value==0) and not noseq: logger.info("ERROR: Some template features are empty")

    return template_features,model2template_mappings

def combine_msas(query_sequences, input_msas, query_cardinality, query_trim, max_seq=None):
    pos=0
    msa_combined=[]

    _blank_seq = [ ("-" * len(seq)) for n, seq in enumerate(query_sequences) for _ in range(query_cardinality[n]) ]

    for n, seq in enumerate(query_sequences):
        for j in range(0, query_cardinality[n]):
            if max_seq: # subsample
                _max_seq = min(max_seq, len(input_msas[n].sequences))
                msa_sample_indices = np.random.choice(len(input_msas[n].sequences), _max_seq, replace=False)
                logger.info(f"     Reducing MSA depth from {len(input_msas[n].sequences)} to {_max_seq}")
            else:
                msa_sample_indices = range(len(input_msas[n].sequences))

            for idx in sorted(msa_sample_indices):
                _desc = input_msas[n].descriptions[idx]
                _seq  = input_msas[n].sequences[idx]

                if not len(set(_seq[query_trim[n][0]:query_trim[n][1]]))>1: continue
                msa_combined.append(">%s"%_desc)
                msa_combined.append("".join(_blank_seq[:pos] + \
                                            [re.sub('[a-z]', '', _seq)[query_trim[n][0]:query_trim[n][1]]] + \
                                            _blank_seq[pos + 1 :]))
            pos += 1


    msas=[pipeline.parsers.parse_a3m("\n".join(msa_combined))]

    return msas





def runme(msa_filenames,
          query_cardinality =   [1,0],
          query_trim        =   [[0,10000],[0,10000]],
          template_fn_list  =   None,
          preds_per_model   =   1,
          jobname           =   'test',
          data_dir          =   '/scratch/AlphaFold_DBs/2.3.2',
          num_recycle       =   3,
          chain_ids         =   None,
          dryrun            =   False,
          do_relax          =   False,
          max_seq           =   None,
          random_seed       =   None,
          nomerge           =   False,
          noseq             =   False,
          truncate          =   None,
          rotrans           =   None,
          pbty_cutoff       =   0.8,
          plddt_cutoff      =   None,
          debug             =   False,
          iterate           =   1,
          fixed_chain_ids   =   None,
          keepalldata       =   False):




    logger.info(" --> Combining input MSAs...")
    msas=[]
    for ia3m, a3m_fn in enumerate(msa_filenames):
        logger.info(f"     #{ia3m} {a3m_fn}")
        with open(a3m_fn, 'r') as fin:
            msas.append(pipeline.parsers.parse_a3m(fin.read()))


    query_sequences=[_m.sequences[0][query_trim[_i][0]:query_trim[_i][1]] for _i,_m in enumerate(msas)]
    query_seq_extended=[_m.sequences[0][query_trim[_i][0]:query_trim[_i][1]] \
                                for _i,_m in enumerate(msas) \
                                for _ in range(query_cardinality[_i])]

    query_seq_combined="".join(query_seq_extended)

    msas = combine_msas(query_sequences, msas, query_cardinality, query_trim, max_seq=max_seq)

    #reproduce af2-like output paths
    # do not clean jobpath - processed template will be stored there before job is started
    jobpath=Path(jobname)
    inputpath=Path(jobname, "input")
    outputpath=Path(jobname, "output")
    msaspath=Path(jobname, "input", "msas", "A")
    for dd in [inputpath, outputpath, msaspath]:
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

    input_template_fn_list = list(template_fn_list)

    logger.info("")
    logger.info(f" --> Combined target sequence:")
    pretty_sequence_print(name_a="        ", seq_a=query_seq_combined)
    logger.info("")

    try:
        rotmax,transmax = map(float, rotrans.split(','))
        logger.info(f"WARNING: Protein chains will be randomly rotated/translated abput their COMs up to {rotmax} deg and {transmax} A")
    except:
        rotmax,transmax=None,None

    if nomerge:
        template_fn_list = template_preps_nomerge_bio(template_fn_list,
                                                  chain_ids,
                                                  target_sequences  =   query_seq_extended,
                                                  outpath           =   inputpath,
                                                  truncate          =   truncate,
                                                  plddt_cutoff      =   plddt_cutoff,
                                                  rotmax            =   rotmax,
                                                  transmax          =   transmax,
                                                  fixed_chain_ids   =   fixed_chain_ids)
        template2input_mapping = None
    else:
        template_fn_list, template2input_mapping = template_preps_bio(template_fn_list,
                                                                      chain_ids,
                                                                      target_sequences  =   query_seq_extended,
                                                                      outpath           =   inputpath,
                                                                      truncate          =   truncate,
                                                                      plddt_cutoff      =   plddt_cutoff,
                                                                      rotmax            =   rotmax,
                                                                      transmax          =   transmax,
                                                                      fixed_chain_ids   =   fixed_chain_ids)

    with tempfile.TemporaryDirectory() as tmp_path:
        template_features,model2template_mappings = generate_template_features(query_sequence   =   query_seq_combined,
                                                                               db_path          =   tmp_path,
                                                                               template_fn_list =   template_fn_list,
                                                                               nomerge          =   nomerge,
                                                                               dryrun           =   dryrun,
                                                                               noseq            =   noseq,
                                                                               debug            =   debug)

    with Path(inputpath, 'mappings.json').open('w') as of:
        of.write(json.dumps({'template2input_mapping':template2input_mapping, 'model2template_mappings':model2template_mappings}))

    model_params = {}
    model_runner_1 = None
    model_runner_3 = None
    for model_idx in range(1,3) if template_fn_list else range(1,6):
        for run_idx in range(1, preds_per_model+1):
            model_name=f"model_{model_idx}"
            if model_name not in list(model_params.keys()):
                model_name_local = f"{model_name}_run{run_idx}"

                if 0:#not Path(data_dir, 'params', 'params_model_1_ptm.npz').exists():
                    suffix=''
                else:
                    suffix='_ptm'

                if data_dir is None:
                    data_dir = Path(os.path.dirname(__file__), '..', 'alphafold', 'data')

                model_params[model_name_local] = data.get_model_haiku_params(model_name=model_name+suffix, data_dir=data_dir)

                if model_idx == 1:
                    model_config = config.model_config(model_name+suffix)
                    model_config.data.common.num_recycle = num_recycle
                    model_config.model.num_recycle = num_recycle
                    model_config.data.eval.num_ensemble = 1
                    model_runner_1 = model.RunModel(model_config, model_params[model_name_local])
                if model_idx == 3:
                    model_config = config.model_config(model_name+suffix)
                    model_config.data.common.num_recycle = num_recycle
                    model_config.model.num_recycle = num_recycle
                    model_config.data.eval.num_ensemble = 1
                    model_runner_3 = model.RunModel(model_config, model_params[model_name_local])

    # gather features
    feature_dict = {
        **pipeline.make_sequence_features(sequence=query_seq_combined, description="none", num_res=len(query_seq_combined)),
        **pipeline.make_msa_features(msas=msas),
        **template_features
    }

    del msas

    feature_dict["asym_id"] = \
            np.array( [int(n+1) for n, l in enumerate(tuple(map(len, query_seq_extended))) for _ in range(0, l)] )
    feature_dict['assembly_num_chains']=len(query_seq_extended)
    with Path(outputpath, 'features.pkl').open('wb') as of: pickle.dump(feature_dict, of, protocol=pickle.HIGHEST_PROTOCOL)

    predict_structure(jobname, query_seq_combined, feature_dict,
                      Ls                        =   tuple(map(len, query_seq_extended)),
                      model_params              =   model_params,
                      model_runner_1            =   model_runner_1,
                      model_runner_3            =   model_runner_3,
                      do_relax                  =   do_relax,
                      model2template_mappings   =   model2template_mappings,
                      random_seed               =   random_seed,
                      template_fn_list          =   input_template_fn_list)

    if PLOTTER_AVAILABLE:
        make_figures(jobname, keepalldata=keepalldata, pbty_cutoff=pbty_cutoff)

    make_contact_scripts(jobname, feature_dict, keepalldata=keepalldata, pbty_cutoff=pbty_cutoff)


def main():

    header_msg = "\n".join(["", f"## gapTrick version {version.__version__}", ""," ==> Command line: gapTrick %s" % (" ".join(sys.argv[1:])), ""])

    start_time = datetime.now()

    (parser, options) = parse_args()

    if options.expert:
        (parser, options) = parse_args(expert = options.expert)
        parser.print_help()
        exit(0)


    if options.jobname is None:
        print( header_msg )
        print('Define jobname - output directory with --jobname')
        exit(0)

    jobpath=Path(options.jobname)
    try:
        jobpath.mkdir(parents=True, exist_ok=False)
    except:
        print( header_msg )
        print(f"ERROR: target directory already exists '{jobpath}'")
        return 1

    logging.basicConfig(level=logging.INFO, format="%(message)s",\
            handlers=[logging.FileHandler(Path(jobpath, 'logfile.txt')),logging.StreamHandler(sys.stdout)])

    logger.info(header_msg)

    if options.msa:

        msas = options.msa.split(',')

    elif options.seqin:
        mmseqspath=Path(options.jobname, "msa")
        mmseqspath.mkdir(parents=True, exist_ok=False)


        existing_msas={}
        if options.msa_dir:
            # create msa_dir, if needed 
            Path(options.msa_dir).mkdir(parents=True, exist_ok=True)

            for fn in glob.glob( os.path.join(options.msa_dir, '*.*') ):
                with open(fn) as ifile:
                    _=ifile.readline()
                    existing_msas[ifile.readline().strip()]=fn
            logger.info(f" --> Parsed {len(existing_msas)} MSA files")
            logger.info("")

        msas = []
        local_msa_dict = {}

        with open(options.seqin) as ifile:
            for record in SeqIO.parse(ifile, "fasta"):
                a3m_fname = existing_msas.get(record.seq, None)
                if not a3m_fname: a3m_fname=local_msa_dict.get(record.seq, None)

                if a3m_fname:
                    logger.info(f" --> Found existing MSA for target sequence [{record.id}]")
                else:
                    if options.msa_dir:
                        a3m_fname = os.path.join(options.msa_dir, f"{uuid.uuid4().hex}.a3m")
                    else:
                        a3m_fname = os.path.join(options.jobname, "msa", f"{len(local_msa_dict):04d}.a3m")

                    query_mmseqs2(record.seq, a3m_fname)
                    local_msa_dict[record.seq]=a3m_fname

                msas.append(a3m_fname)
        logger.info("")
    else:
        logger.info("ERROR: --msa or --seqin required on input")
        exit(1)

    if not options.trim:
        trim = [[0,9999]]*len(msas)
    else:
        trim=[tuple(map(int, _.split(":"))) for _ in options.trim.split(",")]

    if not options.cardinality:
        cardinality = [1]*len(msas)
    else:
        cardinality = tuple(map(int,options.cardinality.split(',')))


    runme(msa_filenames     =   msas,
          query_cardinality =   cardinality,
          query_trim        =   trim,
          preds_per_model   =   options.preds_per_model,
          template_fn_list  =   options.templates.split(',') if options.templates else [],
          jobname           =   options.jobname,
          data_dir          =   options.data_dir,
          num_recycle       =   options.num_recycle,
          chain_ids         =   options.chain_ids,
          dryrun            =   options.dryrun,
          do_relax          =   options.relax,
          max_seq           =   options.max_seq,
          random_seed       =   options.seed,
          nomerge           =   options.nomerge,
          noseq             =   options.noseq,
          truncate          =   options.truncate,
          rotrans           =   options.rotrans,
          pbty_cutoff       =   options.pbty_cutoff,
          plddt_cutoff      =   options.plddt_cutoff,
          debug             =   options.debug,
          iterate           =   options.iterate,
          fixed_chain_ids   =   options.fixed_chain_ids,
          keepalldata       =   options.keepalldata)

    if not options.keepalldata:
        for fname in os.listdir(Path(options.jobname, "msa")):
            if Path(fname).suffix == ".a3m":
                os.remove(Path(options.jobname, "msa", fname))

        for fname in os.listdir(Path(options.jobname, "input")):
            if Path(fname).suffix == ".pkl":
                os.remove(Path(options.jobname, "input", fname))

    logger.info("")
    td = (datetime.now() - start_time) 
    logger.info("Elapsed time %02i:%02i:%02i" % (td.total_seconds()//3600,
                                          (td.total_seconds()%3600)//60,
                                           td.total_seconds()%60))
    logger.info("")
    logger.info(f"Normal termination at {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
    logger.info("")

if __name__=="__main__":
    main()
