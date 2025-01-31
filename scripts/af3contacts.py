import json
import numpy as np
import string
import os,sys,re,io
letters=string.ascii_uppercase+string.ascii_lowercase

from Bio.PDB import PDBIO, PDBParser, Superimposer, MMCIFParser, Select


pymol_header=f"load %(modelid)s.cif\nshow_as cartoon, %(modelid)s\nset label_size, 0\nutil.cbc %(modelid)s"
pymol_dist_generic="""\
dist \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s and name \"%(A_atom_name)s\" and alt \'\', \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s and name \"%(B_atom_name)s\" and alt \'\'"""

tgo = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'O': 'PYL', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'U': 'SEC', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR', 'X': 'UNK'}
ogt = dict([(tgo[_k], _k) for _k in tgo])


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
            pdbio = PDBIO()
            pdbio.set_structure(structure)
            pdbio.save(outstr, select=NotAlt())
            outstr.seek(0)

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(outid, outstr)[0]
            for chain in structure:
                for resi in chain:
                    for atom in resi:
                        atom.set_altloc(" ")

    return structure



if __name__=="__main__":

    if not len(sys.argv)==3:
        print('Usage: [fold_xyz_full_data_0.json] [fold_xyz_model_0.cif]')
        exit(0)


    with open(sys.argv[1], 'r') as ifile:
        dd = json.load(ifile)

    chain_ids = dd['token_chain_ids']
    res_ids = dd['token_res_ids']
    below8pbty = np.array(dd['contact_probs'])

    ref_ph_fn = sys.argv[2]
    print(ref_ph_fn)

    structure = parse_pdb_bio(sys.argv[2], outid="XYZ", remove_alt_confs=True)
    chain_seq_dict = {}
    for chain in structure:
        chain_seq_dict[chain.id]=[_r.get_resname() for _r in chain.get_unpacked_list()]

    chain_lens = [len(_) for _ in chain_seq_dict.values()]
    chain_resid_shifts = {}
    for idx, cid in enumerate(chain_seq_dict):
        chain_resid_shifts[cid] = sum(chain_lens[:idx])

    resi_i,resi_j = np.where(below8pbty>0.8)
    visited = []

    d={'modelid':os.path.basename(ref_ph_fn)[:-4], 'A_atom_name':'CA', 'B_atom_name':'CA'}
    pymol_str= [pymol_header%d]

    for i,j in zip(resi_i, resi_j):
        if chain_ids[i]==chain_ids[j]: continue

        resni = chain_seq_dict[chain_ids[i]][int(i)-1-chain_resid_shifts[chain_ids[i]]].upper()
        resnj = chain_seq_dict[chain_ids[j]][int(j)-1-chain_resid_shifts[chain_ids[j]]].upper()

        #print(f"{res_ids[i]}/{chain_ids[i]} : {res_ids[j]}/{chain_ids[j]} {below8pbty[i,j]:5.2f}")
        print(f"{resni:3s}/{chain_ids[i]}/{str(res_ids[i]):4s} {resnj:3s}/{chain_ids[j]}/{str(res_ids[j]):4s} {below8pbty[i,j]:5.2f}")

        if set([i, j]) in visited: continue
        visited.append( set([i, j]) )
        if resnj in ogt:
            B_atom_name='CA'
        else:
            B_atom_name='P'

        if resni in ogt:
            A_atom_name='CA'
        else:
            A_atom_name='P'

        d = {'modelid':os.path.basename(ref_ph_fn)[:-4],
             'A_atom_name':A_atom_name, 'B_atom_name':B_atom_name,'A_chain':chain_ids[i] ,
             'A_resid':res_ids[i], 'B_chain':chain_ids[j], 'B_resid':res_ids[j]}

        pymol_str.append(pymol_dist_generic%d)
        pymol_str.append(f"show sticks, (chain {d['A_chain']} and resi {d['A_resid']})")
        pymol_str.append(f"show sticks, (chain {d['B_chain']} and resi {d['B_resid']})")

    with open(ref_ph_fn[:-4]+'.pml', 'w') as ofile:
        ofile.write( "\n".join(pymol_str) )
