
__author__ = "Grzegorz Chojnowski"
__date__ = "255555 2022"

#https://mmcif.pdbj.org/converter/index.php?l=en


import sys, os, re

import tempfile
import subprocess
import iotbx
import iotbx.pdb
from cctbx  import maptbx
from cctbx.array_family import flex
from iotbx import ccp4_map

from mmtbx.alignment import align
from iotbx.bioinformatics import any_sequence_format

CLUTALW_SH="""
cd %(tmpdirname)s
clustalw2 input.fa
"""

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


def read_ph(ifname, verbose=True):

    if verbose: print( " ==> Parsing a PDB/mmCIF file: %s" % ifname )


    with open(ifname, 'r') as ifile:
        ph, symm = parse_pdbstring(ifile.read())


    ph.remove_alt_confs(True)

    return ph, symm

def clustal_si(seq_string, refseq_string):

    ccp4 = os.environ.get('CCP4', None)

    with tempfile.TemporaryDirectory(prefix="guessmysequence_refseq_") as tmpdirname:
        with open(os.path.join(tmpdirname, 'input.fa'), 'w') as ofile:
            ofile.write(">1\n%s\n"%seq_string)
            ofile.write(">2\n%s"%refseq_string)

        clustalw_script = CLUTALW_SH%locals()

        ppipe = subprocess.Popen( clustalw_script,
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)

        seq_identity=None
        for stdout_line in iter(ppipe.stdout.readline, ""):
            match = re.search(r'Sequences.*Aligned.*Score:\s*(\d*)', stdout_line)
            if match: seq_identity = int(match.group(1))

        retcode = subprocess.Popen.wait(ppipe)

        return seq_identity


def main(resi_shift=1000):
    fn = 'ddcbe_template.fasta'
    fn = "mycp3_template.fasta"
    fn = 'bbbmm_template.fasta'
    fn = 'bb_template_7npr.fasta'
    fn = 'b2p1_template.fasta'
    fn = 'b1p1_template.fasta'
    fn = 'bbb_template_7npr.fasta'

    with open(fn, 'r') as ifile:
        fasta_obj, _err = any_sequence_format(file_name="wird.fasta", data=ifile.read())

    for _seq in fasta_obj:
        print(_seq.name)

    refseq='PQAAVVAIMAADVQIAVVLDAHAPISVMIDPLLKVVNTRLRELGVAPLEAKGRGRWMLCLVDGTPLRPNLSLTEQEVYDGDRLWLKFLEDTEHRSEVIEHISTAVATNLSKRFAPIDPVVAVQVGATMVAVGVLLGSALLGWWRWQHESWLPAPFAAVIAVLVLTVATMILARSKTVPDRRVGDILLLSGLVPLAVAIAATAPGPVGAPHAVLGFGVFGVAAMLVMRFTGRRLGVYTALVTLCAAATAAGLARMVLLTSAVTLLTCVLLACVLMYHGAPALSRWLSGIRLPVFPSATSRWVFEARPLEGPASVRDVLLRAERARSFLTGLLVGLGVLTVVCLAGLCDPHAGRRWLPLLLAAFTFGFLILRGRSYVDRWQAITLAATAVLIIAAVAVRYVLVSGSPAVLSAGVAVLVLLPAAGLTA'

    ph,symm = read_ph(['7npr_protomer.cif', '6sgx.cif','7b9f.pdb','../esxn/7npr/mycp.pdb.cif', '../esxn/7npr/m2p3_block.cif', '../esxn/7npr/eccBx2.cif', '../esxn/bbp/b1p1.cif', '../esxn/bbp/b2p1.cif', '../esxn/7npr/bbb3.cif'][-1])

    selected_chains={}
    selected_chain_objs=[]

    for tpl_seq in fasta_obj:

        chains_si=[]
        for ch in ph.models()[0].chains():
            print(ch.id)
            if ch.id in selected_chains.values(): continue
            print(selected_chains.values())
            chseq = "".join(ch.as_sequence())

            align_obj = align(seq_b = chseq,
                              seq_a = tpl_seq.sequence, similarity_function="blosum50")

            alignment = align_obj.extract_alignment()
            si = 100*alignment.calculate_sequence_identity()
            si = clustal_si(chseq, tpl_seq.sequence)
            chains_si.append( [si,ch.id,ch] )

            #alignment.pretty_print(block_size=100, top_name="  model", bottom_name="  refseq", show_ruler=False)
        #print(sorted(chains_si, key=lambda x: x[0], reverse=True))
        sorted_chains = sorted(chains_si, key=lambda x: x[0], reverse=True)
        selected_chains[tpl_seq.name]=sorted_chains[0][1]
        selected_chain_objs.append([tpl_seq.name, sorted_chains[0][-1]])


    for _n,_c in selected_chain_objs:
        print(_n.split('=')[-1], _c.id)

    tmp_ph = iotbx.pdb.hierarchy.root()
    tmp_ph.append_model(iotbx.pdb.hierarchy.model(id="0"))
    tmp_ph.models()[0].append_chain(iotbx.pdb.hierarchy.chain(id="A"))

    for ich,(tpl_name, ch) in enumerate(selected_chain_objs):
        _trim=int(tpl_name.split('=')[-1])
        for res in ch.detached_copy().residue_groups()[:_trim]:
            res.resseq=ich*resi_shift+res.resseq_as_int()
            tmp_ph.only_chain().append_residue_group( res )

    tmp_ph.write_pdb_file('bbfl_tmp.pdb')



def cut_by_chid(resi_shift=1000):
    selectded_chids='C5,C6,B3,B6,D1,D3'.split(',')

    ph,symm = read_ph('../esxn/7npr/7npr.cif')
    chaindict={}
    for ch in ph.chains():
        chaindict[ch.id]=ch

    # assembly


    tmp_ph = iotbx.pdb.hierarchy.root()
    tmp_ph.append_model(iotbx.pdb.hierarchy.model(id="0"))
    tmp_ph.models()[0].append_chain(iotbx.pdb.hierarchy.chain(id="A"))

    for ich,chid in enumerate(selectded_chids):
        if chid[0]=='B': trim=90
        else: trim=9999
        for res in chaindict[chid].detached_copy().residue_groups()[:trim]:
            res.resseq=ich*resi_shift+res.resseq_as_int()
            tmp_ph.only_chain().append_residue_group( res )

    tmp_ph.write_pdb_file('cbdx_tmp.pdb')




if __name__=="__main__":
    cut_by_chid()
    exit(1)
    main()
