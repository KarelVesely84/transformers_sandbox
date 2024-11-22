#!/bin/env/python3

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Speech2TextProcessor,
    Speech2TextForConditionalGeneration,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from datasets import load_dataset
import torch

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")  # 30 GB !!!

# Wav2Vec2FeatureExtractor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Speech2TextFeatureExtractor
processor2 = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
model2 = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")

# WhisperFeatureExtractor
processor3 = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model3 = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")


breakpoint()




