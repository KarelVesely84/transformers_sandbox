# add dithering to feature extraction

- Note1: With pre-downloaded model, 'dither' must be added manually into `preprocessor_config.json`.
  Otherwise, the passing by `...Processor(..., dither=dither)` will not work.

- Note2: Not adding dithering to `Wav2Vec2FeatureExtractor`, as it does not contain FBANK feature extraction.

## dithering values

`SpeechToTextFeatureExtractor`: the reasonable dithering constant is `4.0`.

`WhisperFeatureExtractor`: the reasonable dithering constant is `1e-4` (i.e. 4 / 2^15).

## results
decoding with pre-existing models (trained without dithering):

`hf_decode_dithering.py` :
- `Speech2TextFeatureExtractor`
- test model `facebook/s2t-small-librispeech-asr`
- test data `patrickvonplaten/librispeech_asr_dummy` (73 utts)

| dither | with torchaudio | without torchaudio |
|--------|-----------------|--------------------|
| 0.0    | 6.17            | 6.17               |
| 1.0    | 6.43            | 6.43               |
|**4.0** | 6.52            | 6.60               |
| 10.0   | 6.95            | 7.30               |
| 100.0  | 7.56            | 7.39               |
| 1000.0 | 38.60           | 37.39              |



`hf_decode_dithering_whisper.py`:
- `WhisperFeatureExtractor`
- test model `openai/whisper-tiny.en`
- test data `patrickvonplaten/librispeech_asr_dummy` (73 utts)

| dither   | with torch |
|----------|------------|
| 0.0      | 22.69      |
|**0.0001**| 22.69      |
| 0.001    | 23.13      |
| 0.01     | 27.56      |
