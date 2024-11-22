#!/usr/bin/bash

exit 0  # use copy-paste

conda create --prefix ${SOME_DIR} python=3.11

conda activate ${SOME_DIR}

conda install pytorch torchvision torchaudio pytorch-cuda=12.1  # with torchaudio
# or
conda install pytorch pytorch-cuda=12.1  # no torchaudio

pip install ipdb jiwer librosa

git clone --branch add_dither git@github.com:KarelVesely84/transformers.git transformers_dither

cd transformers_dither

pip install --editable .

# download existing models and testing data
bash hf_pre-download_data.py

# run the decoding with SpeechToTextFeatureExtractor
bash hf_decode_dithering.sh

# run the decoding with WhisperFeatureExtractor
bash hf_decode_dithering_whisper.sh

