[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gchojnowski/gapTrick/blob/main/gapTrick.ipynb) [![CI](https://github.com/gchojnowski/gapTrick/actions/workflows/test.yml/badge.svg)](https://github.com/gchojnowski/gapTrick/actions)
<!--
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
-->

![contacts_prediction](/examples/contacts.png)

# gapTrick – AlphaFold with multimeric templates

**gapTrick** is a tool based on monomeric AlphaFold2 models that can identify critical residue-residue interactions in low-accuracy models of protein complexes. The approach can aid in the interpretation of challenging cryo-EM and MX structures and the computational identification of protein-protein interactions.

- [Citing this work](#citing-this-work)
- [Colab notebook](#colab-notebook)
- [Installation](#installation)
    - [Hardware requirements](#hardware-requirements)    
    - [Dependencies](#dependencies)
    - [AlphaFold2](#alphafold2)
    - [AlphaFold2 NN parameters](#alphafold2-nn-parameters)
    - [gapTrick](#gaptrick)

- [How to use gapTrick](#how-to-use-gaptrick)
    - [Input](#input)
    - [Running the predictions](#running-the-predictions)
    - [Fair use of the MMseqs2 API](#fair-use-of-the-mmseqs2-api)
    - [Prediction results](#prediction-results)
    - [Troubleshooting](#troubleshooting)
- [Using gapTrick for cryoEM model building](#using-gaptrick-for-cryoEM-model-building)


<br/> 

# Citing this work

Chojnowski bioRxiv (2025) [01.31.635911](https://doi.org/10.1101/2025.01.31.635911)

The gapTrick code depends on [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2) and [MMseqs2](https://doi.org/10.1093/bioinformatics/btab184), remember to cite them as well!

<br/> 

# Colab notebook

You can run gapTrick on Colab server without having to satisfy its hardware (GPU) and software dependencies on your computer. Click the link to start a new Colab session: 
[gapTrick.ipynb](https://colab.research.google.com/github/gchojnowski/gapTrick/blob/main/gapTrick.ipynb)

<br/> 

# Installation

The code requires only a standard AlphaFold2 installation to run. Check AlphaFold2 installation instructions at the [official webpage](https://github.com/google-deepmind/alphafold) or follow the instructions below. I use them to install the code on Colab. It should work smoothly on most Linux distributions.

If you already have a working copy of AlphaFold2 go directly to [gapTrick installation](#gaptrick) instructions. You can also run gapTrick directly from a cloned repository.

## Hardware requirements

gapTrick has the same hardware requirements as AlphaFold2. In most of the cases a standard GPU (T4 on colab, 3090) will be enough as it can handle predictions up to roughly 1,200 residues. For larger targets, you will need to either use cards with more memory (e.g. A100 that can handle up to roughly 3,000 residues), or split your target into smaller fragments.

## Dependencies
First, install the NVIDIA CUDA Toolkit on the system level.
<!--
```
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
```

```
sudo apt-get install cuda-toolkit-12-2
```
-->

Create a target directory, build base conda environment, and install dependencies

```
mkdir gapTrick
cd gapTrick
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p conda
source conda/bin/activate
conda install -qy -c conda-forge -c bioconda python=3.10 openmm=8.0.0 matplotlib kalign2 hhsuite pdbfixer pyopenssl==22.0.0 git
```

## AlphaFold2
Once you installed all dependencies install AlphaFold2 from an official repository

```
git clone --branch main https://github.com/deepmind/alphafold alphafold
pip install -r alphafold/requirements.txt
pip install --no-dependencies ./alphafold
pip install --upgrade "jax[cuda12]"==0.4.26
pip install --upgrade tensorflow==2.16.1
mkdir -p conda/lib/python3.10/site-packages/alphafold/common/
curl -o conda/lib/python3.10/site-packages/alphafold/common/stereo_chemical_props.txt https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
```

## AlphaFold2 NN parameters

To finalize installatin you need to download the AlphaFold2 neural network (NN) parameters and place them in a directory that gapTrick can find (you don't need to download sequence databases)

If you installed gapTrick from scratch, use the following to place the parameters where they can be easily found
```
mkdir --parents conda/lib/python3.10/site-packages/alphafold/data/params
curl -o alphafold_params_2021-07-14.tar https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar
tar --extract --verbose --file=alphafold_params_2021-07-14.tar --directory=conda/lib/python3.10/site-packages/alphafold/data/params --preserve-permissions
rm alphafold_params_2021-07-14.tar
```

If you use an existing installation of AlphaFold2, or prefer to place the parameters in a custom directory (e.g. ``/a/directory/with/parameters``), run the following

```
mkdir --parents /a/directory/with/parameters/params
curl -o /a/directory/with/parameters/params/alphafold_params_2021-07-14.tar https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar
tar --extract --verbose --file=/a/directory/with/parameters/params/alphafold_params_2021-07-14.tar --directory=/a/directory/with/parameters/params --preserve-permissions
rm /a/directory/with/parameters/params/alphafold_params_2021-07-14.tar
```

and use gapTrick with a flag ``--data_dir=/a/directory/with/parameters/``

## gapTrick
The gapTrick itself can be installed with a command:
```
pip install git+https://github.com/gchojnowski/gapTrick
gapTrick --help
```

but you can also use it directly from a cloned repository, e.g. with a shared copy of AlphaFold2 on a cluster. If you use your own installation, remember to specify directory with AF2 parameters using ``--data_dir`` keyword

```
git clone --recursive https://github.com/gchojnowski/gapTrick.git
python gapTrick/gapTrick --help
```
<!--
 ## Installation issues

  - none so far...
<br/> 
-->

# How to use gapTrick

The prediction of protein-protein complexes in gapTrick is based on AlphaFold2 neural network (NN) models trained on single-chain proteins. The method uses only two out of five AF2 NN models that allow for the use of input templates.  To allow prediction of multimeric structural models, all input protein chains are merged into a single protein chain interspersed with 200 amino acid gaps. The only input required from user is a PDB/mmCIF template and a FASTA file containing all target sequences in arbitrary order (it doesn't make sense to run it without a template). The structure prediction is performed fully automatically.

The most important gapTrick outputs are a model and precicted contacts. Two residues are predicted to be in contact if the probability that their CB atoms (CA for glycine) are within 8Å radius is greater than 80%.  **The contact predictions are very precise.** If you see one of them on the complex interface, it's a strong evidence that your complex prediction is correct.

## Input

The most important keywords are

- ``--seqin`` - input sequences in FASTA format (number of chains must match the target, e.g. for a homodimer same sequence must be repeated twice)
- ``--template`` - input template in PDB/mmCIF format
- ``--jobnmame`` - name of a target job directory. Existing directories will not be overwritten

To see a full keywords list run 

```
gapTrick --help
```

## Running the predictions

First, check your installation with a most basic run based on files provided in a examples directory. It's a small homo-dimer, part of a larger complex ([1bjp](https://www.ebi.ac.uk/pdbe/entry/pdb/1bjp/index)) that would be difficult to predict without a template

```
gapTrick --seqin examples/piaq.fasta --template examples/piaq.pdb --jobname piaq_test --max_seq 5000 --relax
```
this will automatically download MSAs from MMseqs2 API and run prediction. Remember about fair use ot the API server.

## Fair use of the MMseqs2 API

gapTrick by default uses MMseqs2 API to generate MSAs. MMseqs2 API is a shared resource that has been generously provided by the developers and as such should be used in a fair manner (in simple words; not TOO extensively). Otherwise your IP **will be blocked** for a while. To reduce the number of requests for predictions repeated for the same target use ``--msa_dir`` keyword. It will store MSA files in a local directory and reuse in subsequent jobs. For example, the following will use a directory ``local_msas``:

```
gapTrick --seqin examples/piaq.fasta --template examples/piaq.pdb --jobname piaq_test --max_seq 5000 --relax --msa_dir local_msas
```

now, whenever you rerun the job above, gapTrick will check the ``local_msas`` directory for MSAs matching your target sequences.

Alternatively, you can pre-calculate target MSAs, store ``*.a3m`` files in a directory and reuse with ``--msa_dir`` keyword. A script ``colabfold_search`` distributed with [ColabFold](https://github.com/sokrypton/ColabFold?tab=readme-ov-file) is a pretty handy solution here. All gapTrick benchmarks were based on MSAs generated with the following command (check [ColabFold website](https://github.com/sokrypton/ColabFold?tab=readme-ov-file#generating-msas-for-large-scale-structurecomplex-predictions) for more details)

```
colabfold_search --db1 uniref30_2202_db --mmseqs /path/to/bin/mmseqs --use-env 0 --filter 0 --db-load-mode 2 input_sequences.fasta /path/to/db_folder msas
```

## Prediction results

After a job finishes the output directory will contain the following files and directories

- ``msas/`` - MSAs downloaded from MMseqs2 API. They can be used in subsequent jobs with the ``--msa_dir`` keyword (unless you haven't used it already).
- ``output/ranked_0.pdb`` - top-ranked prediciton in PDB format.
- ``output/ranked_0_pae.json`` - PAE matrix for top-ranked prediciton. You can use it for generating self-restrains in ISOLDE.
- ``figures/`` - PAE, pLDDT, and distogram plots in png and svg format (created only if matplotlib is installed)
- ``output/contacts.txt`` - list of all residue pairs predicted to be at most 8Å apart with corresponding probabilitiesn (above 0.8). Leading * marks inter-chain intercations. If you have them, the complex prediction is very likely to be correct.
- ``output/pymol_interchain_contacts.pml`` - a pymol script for displaying inter-chain contacts. First, open ranked_0.pdb and then use File->Run Script option to run the scirpt.
- ``output/pymol_interchain_saltbridges.pml`` - this pymol script will add salt-bridges only to your model (if any are detected). These may be the most crucial intercations!
- ``output/chimerax_interchain_contacts.cxc`` - a script for displaying contacts in ChimeraX. Open top prediction and run a command ``run [path to scirpt]/chimerax_interchain_contacts.cxc [model id]``
- ``output/chimerax_interchain_saltbridges.cxc`` - a script for displaying inter-chain salt-bridges in ChimeraX. Open top prediction and run a command ``run [path to scirpt]/chimerax_interchain_saltbridges.cxc [model id]``

## Troubleshooting

 - if running gapTrick produces models identical to those given on input (and there are no contact predictions), it often means that the template is far from a native conformation

# Using gapTrick for cryoEM model building

gapTrick can be very useful in interpreting your cryo-EM (and MX!) maps. Single-chain predictions fittend into maps and rebuilt with gapTrick will have very good stereochemical properties (**easy to refine**) and provide contact predictions (**powerful validation score**), which will help in functinal analysis. Here, in a few simple steps, I will show how I use it

 - Fit your chains (or complexes, if available) into the cryo-EM map. I like to use Molrep in [Doppio](https://www.ccpem.ac.uk/docs/doppio/index.html).
 - Refine the models in real space with self-restraints (COOT or ISOLDE). It needs to fit the map as good as possible.
 - Run gapTrtick on the initial model. If it was close to a native complex, you will see many inter-chain contact predictions. This is a very strong evidence that your model is correct (see the paper). If not, you may need to refine the model further or check your priors (are there any known deleterious mutants at the interface?).
 - Once you are happy with a prediction (you can run gapTrick iteratively), you can proceed with automatic refinement. I like to use [Servalcat](https://servalcat.readthedocs.io/) with half-maps and jelly body restraints (a bit stronger than default at lower resolutions). The gapTrick predictions usually have very good geometry, which should not deteriorate during refinement (use DOPPIO validation tools). Also check the final B-factor distribution. It it's not smooth (should resemble [inverse-gamma distribution](https://doi.org/10.1107/S2059798319004807)) try increasing the initial B-factor in Servalcat. B-factors can be very high in some regions, but that's OK if the map resolution is heterogenous.

(c) 2025 Grzegorz Chojnowski
