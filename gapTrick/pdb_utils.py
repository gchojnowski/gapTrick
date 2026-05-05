

__author__ = "Grzegorz Chojnowski"
__date__ = "5 May 2026"


import os, sys, re, io
from pathlib import Path
from itertools import groupby
from operator import itemgetter
import numpy as np

from Bio import Align
from Bio.PDB import PDBParser, Select, MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.vectors import rotaxis2m
from Bio.PDB.vectors import Vector

tgo = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'O': 'PYL', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'U': 'SEC', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR', 'X': 'UNK'}
ogt = dict([(tgo[_k], _k) for _k in tgo])

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

def match_template_chains_to_target_bio(structure, target_sequences, logger):
    logger.info(f" --> Greedy matching template chains to target sequences")

    chain_seq_dict = {}
    chain_ends_dict = {}
    protein = get_prot_chains_bio(structure, logger)
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

# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

def random_point_on_sphere():
    z = np.random.uniform(-1,1)
    t = 2.0*np.pi * np.random.uniform(0,1);
    r = np.sqrt(1.0-z*z);
    return np.array([r * np.cos(t), r * np.sin(t), z])


# -----------------------------------------------------------------------------

def get_prot_chains_bio(structure, logger, min_prot_content=0.1, truncate=None, rotmax=None, transmax=None, fixed_chain_ids=None):
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

