# gapTrick – structural characterisation of protein-protein interactions using AlphaFold with multimeric templates

**gapTrick** is an approach based on monomeric AlphaFold2 models that can identify critical residue-residue interactions in low-accuracy models of protein complexes. The approach can aid in the interpretation of challenging experimental structures and the computational identification of protein-protein interactions.

- [How to cite](#how-to-cite)
- [Colab notebook](#colab-notebook)
- [Installation](#installation)
    - [Dependencies](#dependencies)
    - [AlphaFold2](#alphafold2)
    - [gapTrick](#gaptrick)
- [How to use gapTrick](#how-to-use-gaptrick)
<br/> 

# How to cite

Grzegorz Chojnowski, to be published.

<br/> 

# Colab notebook

The repository  provides a Colab notebook that can be used to run gapTrick without having to satisfy its hardware (GPU) and software dependencies on your computer. Click the link to start a new Colab session: 
[gapTrick_custom.ipynb](https://colab.research.google.com/github/gchojnowski/gapTrick/blob/main/gapTrick_custom.ipynb)

<br/> 

# Installation

The code requires only a standard AlphaFold2 installation to run. Check AlphaFold2 installation instructions at the [official webpage](https://github.com/google-deepmind/alphafold) or follow the instructions below. I use them to install the code on Colab.

## Dependencies
First create a target directory, create a basic conda environment, and install dependencies

```
mkdir AlphaFold2
cd ALphaFold2
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Darwin-arm64.sh -b -p conda
source conda/bin/activate
conda install -qy conda==24.11.1 \
conda install -qy -c conda-forge -c bioconda python=3.10 openmm=8.0.0 matplotlib kalign2 hhsuite pdbfixer 
```

<!--
```
mkdir AlphaFold2
cd ALphaFold2
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p conda
source conda/bin/activate
conda install -qy conda==24.11.1 \
conda install -qy -c conda-forge -c bioconda python=3.10 openmm=8.0.0 matplotlib kalign2 hhsuite pdbfixer 
```
-->

## AlphaFold2
Once you secured all dependencies install AlphaFold2
```
git clone --branch main https://github.com/deepmind/alphafold alphafold
pip3 install -r alphafold/requirements.txt
pip3 install --no-dependencies alphafold
pip3 install pyopenssl==22.0.0
mkdir -p alphafold/common
curl -o alphafold/common/stereo_chemical_props.txt https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
mkdir -p conda/lib/python3.10/site-packages/alphafold/common/
cp -f alphafold/common/stereo_chemical_props.txt /opt/conda/lib/python3.10/site-packages/alphafold/common/
```

.. and download model weights
```
mkdir --parents alphafold/data/params
curl -O alphafold/data/params/alphafold_params_2021-07-14.tar https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar
tar --extract --verbose --file=alphafold/data/params/alphafold_params_2021-07-14.tar --directory=alphafold/data/params --preserve-permissions
```

## gapTrick
The gapTrick itself is a relatively simple script and is trivial to install with a command:
```
pip install git+https://github.com/gchojnowski/gapTrick
```

<br/> 


# How to use gapTrick

```
gapTrick --help
```
(C) 2025 Grzegorz Chojnowski
