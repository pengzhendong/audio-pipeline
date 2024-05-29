# audio-pipeline

1. pipeline
  - ffmpeg extract
  - pyloudnorm loudnorm
  - silero-vad slice
  - deepfilternet denoise
2. transcribe
  - paraformer
  - cam++ language detection
  - panns audio tags
  - pyannote num of speakers
