# gapTrick – structural characterisation of protein-protein interactions using AlphaFold with multimeric templates

**gapTrick** is a tools based on monomeric AlphaFold2 models that can identify critical residue-residue interactions in low-accuracy models of protein complexes. The approach can aid in the interpretation of challenging experimental structures and the computational identification of protein-protein interactions.

- [How to cite](#how-to-cite)
- [Colab notebook](#colab-notebook)
- [Installation](#installation)
    - [Dependencies](#dependencies)
    - [AlphaFold2](#alphafold2)
    - [gapTrick](#gaptrick)
- [How to use gapTrick](#how-to-use-gaptrick)
    - [Running the predictions](#running-the-predictions)
    - [Interpreting prediction results](#interpreting-prediction-results)


<br/> 

# How to cite

Grzegorz Chojnowski, to be published.

<br/> 

# Colab notebook

You can run gapTrick on Colab server without having to satisfy its hardware (GPU) and software dependencies on your computer. Click the link to start a new Colab session: 
[gapTrick.ipynb](https://colab.research.google.com/github/gchojnowski/gapTrick/blob/main/gapTrick.ipynb)

<br/> 

# Installation

The code requires only a standard AlphaFold2 installation to run. Check AlphaFold2 installation instructions at the [official webpage](https://github.com/google-deepmind/alphafold) or follow the instructions below. I use them to install the code on Colab. It should work smoothly on recent Linux distributions.

If you already have a running AlphaFold2 go directly to [gapTrick installation](#gaptrick) instructions. You can also run gapTrick directly from a cloned repository.

## Dependencies
First, create a target directory, build base conda environment, and install dependencies

```
mkdir AlphaFold2
cd ALphaFold2
curl -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p conda
source conda/bin/activate
conda install -qy -c conda-forge -c bioconda python=3.10 openmm=8.0.0 matplotlib kalign2 hhsuite pdbfixer pyopenssl==22.0.0
```

## AlphaFold2
Once you installed all dependencies install AlphaFold2 from an official repository

```
git clone --branch main https://github.com/deepmind/alphafold alphafold
pip3 install -r alphafold/requirements.txt
pip3 install --no-dependencies alphafold
mkdir -p conda/lib/python3.10/site-packages/alphafold/common/
curl -o conda/lib/python3.10/site-packages/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
```

... and download model weights
```
mkdir --parents alphafold/data/params
curl -O alphafold/data/params/alphafold_params_2021-07-14.tar https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar
tar --extract --verbose --file=alphafold/data/params/alphafold_params_2021-07-14.tar --directory=alphafold/data/params --preserve-permissions
```

## gapTrick
The gapTrick itself can be installed with a command:
```
pip install git+https://github.com/gchojnowski/gapTrick
```

<br/> 


# How to use gapTrick

## Running the predictions
First, check your installation with a most basic run based on files provided in a examples directory. It's a part of a larger compels ([1bjp](https://www.ebi.ac.uk/pdbe/entry/pdb/1bjp/index)) that would be difficult to predict without a template
```
gapTrick --seqin examples/piaq.fasta --templates examples/1bjp2.pdb --jobname piaq_test --max_seq 5000 --relax
```
this will automatically download MSAs from MMseqs2 API and run prediction. Remember about fair use ot the API server. To reduce the number of requests for predictions repeated for teh same target use ``--msa_dir`` keyword. It will store MSA files in a local directory and reuse in later jobs. For example, the following will use a directory ``local_msas`` (you need to create it in advance):

```
gapTrick --seqin examples/piaq.fasta --templates examples/piaq.pdb --jobname piaq_test --max_seq 5000 --relax --msa_dir local_msas
```
now, whenever you rerun the job above gapTrick will check the ``local_msas`` directory for MSAs matching your target sequences

## Interpreting prediction results

After a job finishes the outpout directory will contain the following files and directories

    - msas
    - input/ranked_0.pdb - top-ranked prediciton
    - input/ranked_0_pae.json - PAE matrix for top-ranked prediciton. You can use it for generating self-restrains in ISOLDE.
    - figures/ - PAE, pLDDT, and distogram plots in png and svg format
    - contacts.txt - list of all residue pairs predicted to be at most 8Å apart with corresponding probabilities. Leading * mark inter-chain ones, if you have them, the complex prediction is most likely correct
    - pymol_interchain_contacts.pml - a pymol script for displaying inter-chain contacts (first, open ranked_0.pdb and then use File->Run Script option to run it).
    - pymol_all_contacts.pml - same as above with all cointacts, there is ususally lots of them!

(C) 2025 Grzegorz Chojnowski
