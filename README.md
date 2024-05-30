# audio-pipeline

1. pipeline
  - ffmpeg extract
  - pyloudnorm loudnorm
  - silero-vad slice
  - deepfilternet denoise
2. transcribe
  - paraformer asr
  - cam++ language detection
  - pyannote num of speakers
  - panns audio tags
