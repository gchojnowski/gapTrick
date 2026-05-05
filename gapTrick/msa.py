
__author__ = "Grzegorz Chojnowski"
__date__ = "5 May 2026"

import os
import time
import tempfile
import requests
import tarfile
import json


# -----------------------------------------------------------------------------

def pretty_sequence_print(name_a, seq_a, logger, name_b=None, seq_b=None, block_width=80):

    #if seq_b: assert len(seq_a) == len(seq_b)

    length = len(seq_a)
    n_blocks = length//block_width

    for ii in range(n_blocks+1):
        logger.info(f"{name_a} {seq_a[ii*block_width:(ii+1)*block_width]}")
        if seq_b:
            logger.info(f"{name_b} {seq_b[ii*block_width:(ii+1)*block_width]}")
            logger.info("")

# -----------------------------------------------------------------------------

def query_mmseqs2(query_sequence, msa_fname, mmseqs_api_server, logger, use_env=False, filter=False, user_agent='gaptrick'):

    def submit(query_sequence, mode):
        while True:
            try:
                res = requests.post(f'{mmseqs_api_server}/ticket/msa', data={'q':f">1\n{query_sequence}", 'mode': mode}, timeout=12.01, headers=headers)
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
                res = requests.get(f'{mmseqs_api_server}/ticket/{ID}', timeout=12.01, headers=headers)
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
                res = requests.get(f'{mmseqs_api_server}/result/download/{ID}', timeout=12.01, headers=headers)
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
                logger.error("ERROR: MMseqs2 API request rejected (too many connections). Try again later...")
                exit(1)

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


