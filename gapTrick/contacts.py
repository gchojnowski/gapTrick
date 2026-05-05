

__author__ = "Grzegorz Chojnowski"
__date__ = "5 May 2026"


from pathlib import Path
import glob
import os
import pickle
import numpy as np
import string

from gapTrick.pdb_utils import parse_pdb_bio, get_prot_chains_bio

# templates for a pymol script visualising predficted contacts
pymol_dist_generic="""\
dist \"%(modelid)s\" and chain \"%(A_chain)s\" and resi %(A_resid)s and name \"%(A_atom_name)s\" and alt \'\', \"%(modelid)s\" and chain \"%(B_chain)s\" and resi %(B_resid)s and name \"%(B_atom_name)s\" and alt \'\'"""

pymol_header=f"load %(modelid)s.pdb\nshow_as cartoon, %(modelid)s\nset label_size, 0\nutil.cbc %(modelid)s"

chimerax_footer="distance style radius 0.15\ndistance style color red\ndistance style dashes 0\ncolor bychain"
chimerax_dist_generic=\
        "\n".join(["distance #$1/%(A_chain)s:%(A_resid)s@%(A_atom_name)s #$1/%(B_chain)s:%(B_resid)s@%(B_atom_name)s",
                   "show #$1/%(A_chain)s:%(A_resid)s bonds",
                   "show #$1/%(B_chain)s:%(B_resid)s bonds"])


# lists likely contacts and generates pymol/chimera scripts
# bypasses af2plots and has no matplolib dep

def make_contact_scripts(prefix, feature_dict, logger, print_contacts=False, keepalldata=False, pbty_cutoff=0.8, distance_cutoff=8.0):

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
    #logger.info(f"AlphaFold2 distogram distance range [{bin_edges[0]}, {bin_edges[-1]}]")

    # truncate distance to the available range
    distance = np.clip(distance_cutoff, 3, 20)

    bin_idx=np.max(np.where(bin_edges<distance))


    below8pbty = np.sum(probs, axis=2, where=(np.arange(probs.shape[-1])<bin_idx))

    requested_contacts=[]
    if print_contacts:
        logger.info()
        logger.info(f"AlphaFold2-predicted contacts below {distance}A with estimated probability (*-inter chains)")

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

        if print_contacts: logger.info(f"{'*' if ci!=cj else ' '} {reli:-4d}/{chain_ids[ci]} {relj:-4d}/{chain_ids[cj]} {below8pbty[i,j]:5.2f}")

    # contacts list
    contact_template = r"^(?P<res1>\w+?)/(?P<ch1>\w+?)\s+(?P<res2>\w+?)/(?P<ch2>\w+?)\s+(?P<pbty>[\d\.]*?)$"
    structure = parse_pdb_bio(Path(prefix, "output", "ranked_0.pdb"), outid="XYZ", remove_alt_confs=True)
    protein = get_prot_chains_bio(structure, logger)
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
        ofile.write("residue_1 residue_2 pbty(|CB-CB|<8Å)>0.8\n")
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

