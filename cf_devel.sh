#!/bin/bash

root='/cryo_em/AlphaFold/scripts/af2_multimers_with_templates'

export MODULEPATH=/opt/ohpc/pub/modulefiles:/g/easybuild/x86_64/CentOS/7/haswell/modules/all
. /opt/ohpc/admin/lmod/lmod/init/bash

module load AlphaFold
module load GCC/10.2.0
module load tqdm
module load matplotlib

export DEVEL_DIR=/cryo_em/AlphaFold/ColabFold/cf_devel
export PYTHONPATH=$DEVEL_DIR/site-packages:$PYTHONPATH
export PYTHONPATH=$DEVEL_DIR/site-packages/alphafold:$PYTHONPATH


#python -m colabfold.batch --data /cryo_em/AlphaFold/DBs $@
#$DEVEL_DIR/site-packages/bin/colabfold_batch --data /cryo_em/AlphaFold/DBs $@
TF_FORCE_UNIFIED_MEMORY='1' XLA_PYTHON_CLIENT_MEM_FRACTION='4.0' python3 $root/cf_custom.py $@
