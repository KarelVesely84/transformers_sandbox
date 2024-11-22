#!/usr/bin/bash

# make 'conda activate' findable:
[ -z "${CONDA_EXE}" ] && echo "Error, missing $CONDA_EXE !" && exit 1
CONDA_BASE=$(${CONDA_EXE} info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate /mnt/matylda5/iveselyk/ASR_TOOLKITS/K2_SHERPA_PYTORCH24_CUDA121/TRANSFORMERS_TORCHAUDIO

set -euxo pipefail

export CUDA_VISIBLE_DEVICES=$(free-gpu.py)

python3 hf_decode_dithering_whisper.py 0.0001
