#!/usr/bin/env python3

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

from datasets import load_dataset
import torch

from jiwer import wer


# cli:
import sys
dither, = sys.argv[1:]
dither = float(dither)
print(f"### hf_decode_dithering.py dither {dither}")

# load model and tokenizer

# Note:
# "dither" must be added into "preprocessor_config.json", to make the Processor override work.

# WhisperFeatureExtractor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en", dither=dither)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to("cuda")

# load super-small dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

def map_to_pred(batch):
    sampling_rate = batch['audio'][0]['sampling_rate']
    inputs = processor(batch["audio"][0]["array"], sampling_rate=sampling_rate, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"].to("cuda"))

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    transcription = [ t.upper() for t in transcription ]
    batch["transcription"] = transcription
    return batch

result = ds.map(map_to_pred, batched=True, batch_size=1, remove_columns=["audio"])

print(f"reference {result['text']}")
print(f"asr_output {result['transcription']}")

print("WER:", wer(result["text"], result["transcription"]))

