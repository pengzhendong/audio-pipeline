# from funasr import AutoModel
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks

# inference_pipeline = pipeline(
#     task=Tasks.auto_speech_recognition,
#     model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
#     model_revision="v2.0.4",
#     vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
#     vad_model_revision="v2.0.4"
# )

# rec_result = inference_pipeline(
#     "/usr/pengzhendong/data/original/kaini/loudnorm/raw_0122/举报标准1.wav"
# )

# data = rec_result[0]
# print(data)
# print(data["text"], len(data["text"]))

# model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
# res = model.generate(input="/usr/pengzhendong/data/original/kaini/loudnorm/raw_0122/举报标准1.wav")
# print(res)

# model = AutoModel(model="ct-punc", model_revision="v2.0.4")
# res = model.generate(input=data["text"])
# print(res)


import librosa
import numpy as np
import soundfile as sf

for line in open("wav.scp"):
    wav = line.strip()
    audio, sr = sf.read(wav, dtype=np.float32)
    freqs = librosa.fft_frequencies(sr=sr)
    freq_band_idx = np.where(freqs >= 17500)[0]
    energy = np.sum(np.abs(librosa.stft(audio)[freq_band_idx, :]))
    print(wav, energy)
