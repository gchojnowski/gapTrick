import json
import numpy as np
import string
import os,sys,re
letters=string.ascii_uppercase+string.ascii_lowercase



pymol_header=f"load %(modelid)s.cif\nshow_as cartoon, %(modelid)s\nset label_size, 0\nutil.cbc %(modelid)s"
pymol_dist_generic="""\
dist \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s and name \"%(A_atom_name)s\" and alt \'\', \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s and name \"%(B_atom_name)s\" and alt \'\'"""






if __name__=="__main__":

    if not len(sys.argv)==3:
        print('Usage: [json file] [cif file]')
        exit(0)


    with open(sys.argv[1], 'r') as ifile:
        dd = json.load(ifile)

    chain_ids = dd['token_chain_ids']
    res_ids = dd['token_res_ids']
    below8pbty = np.array(dd['contact_probs'])

    ref_ph_fn = sys.argv[2]
    print(ref_ph_fn)

    resi_i,resi_j = np.where(below8pbty>0.5)
    visited = []

    d={'modelid':os.path.basename(ref_ph_fn)[:-4], 'A_atom_name':'CA', 'B_atom_name':'CA'}
    pymol_str= [pymol_header%d]

    for i,j in zip(resi_i, resi_j):
        if chain_ids[i]==chain_ids[j]: continue


        print(f"{res_ids[i]}/{chain_ids[i]} : {res_ids[j]}/{chain_ids[j]} {below8pbty[i,j]:5.2f}")


        if set([i, j]) in visited: continue
        visited.append( set([i, j]) )


        d = {'modelid':os.path.basename(ref_ph_fn)[:-4],
             'A_atom_name':'CA', 'B_atom_name':'CA','A_chain':chain_ids[i] ,
             'A_resid':res_ids[i], 'B_chain':chain_ids[j], 'B_resid':res_ids[j]}

        pymol_str.append(pymol_dist_generic%d)
        pymol_str.append(f"show sticks, (chain {d['A_chain']} and resi {d['A_resid']})")
        pymol_str.append(f"show sticks, (chain {d['B_chain']} and resi {d['B_resid']})")

    with open(ref_ph_fn[:-4]+'.pml', 'w') as ofile:
        ofile.write( "\n".join(pymol_str) )
