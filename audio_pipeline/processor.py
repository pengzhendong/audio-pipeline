# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from functools import partial
from pathlib import Path

import ffmpeg
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from df.enhance import enhance, init_df, load_audio, save_audio
from funasr import AutoModel
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from panns_inference import AudioTagging, labels
from pyannote_onnx import PyannoteONNX
from silero_vad import SileroVAD
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


from utils import format_exception


df_repo_dir = snapshot_download("pengzhendong/DeepFilterNet")
panns_repo_dir = snapshot_download("pengzhendong/panns")
panns_labels = []
for label in labels:
    label = label.lower().split(", ")[0].replace(" ", "_")
    panns_labels.append(label)


def extract(in_path, out_path, stream=0, channel=0, sampling_rate=None):
    """
    Extract a mono audio from a multi-channel audio or video file.

    Args:
        in_path: A multi-channel audio or video file.
        out_path: The output mono audio file.
        stream: Stream index of the audio to extract. Defaults to 0.
        channel: Channel index of the stream to extract. Defaults to 0.
        sampling_rate: Sampling rate of the outputs.
    """
    # https://trac.ffmpeg.org/wiki/audio%20types
    # https://trac.ffmpeg.org/wiki/AudioChannelManipulation
    try:
        stream = ffmpeg.input(in_path)[f"a:{stream}"]
        stream = ffmpeg.filter(stream, "pan", **{"mono|c0": f"c{channel}"})
        if sampling_rate is not None:
            stream = ffmpeg.filter(stream, "aresample", sampling_rate)
        stream = ffmpeg.output(stream, str(out_path), loglevel="quiet", stats=None)
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)
    except Exception as e:
        with open(Path(out_path).with_suffix(".log"), "w+") as fout:
            fout.write(f"{in_path}\n{format_exception(e)}\n")


def loudnorm(in_path, out_path, peak=-1.0, loudness=-23.0, block_size=0.400):
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        in_wav: input audio file.
        out_wav: output audio file.
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)
    """
    try:
        audio, sample_rate = sf.read(in_path)
        # peak normalize audio to [peak] dB
        audio = pyln.normalize.peak(audio, peak)
        # measure the loudness first (BS.1770 meter)
        loudness = pyln.Meter(sample_rate, block_size=block_size).integrated_loudness(
            audio
        )
        audio = pyln.normalize.loudness(audio, loudness, loudness)
        sf.write(out_path, audio, sample_rate)
    except Exception as e:
        with open(Path(out_path).with_suffix(".log"), "w+") as fout:
            fout.write(f"{in_path}\n{format_exception(e)}\n")


def vad(
    in_path, out_json, save_slices=False, speech_pad_ms=100, min_silence_duration_ms=200
):
    """
    Perform voice activity detection (VAD) on an audio file.

    Args:
        in_path: input audio file.
        out_json: output json file.
        save_slices: whether to save the slices of the audio.
        speech_pad_ms: final speech chunks are padded by speech_pad_ms each side.
        min_silence_duration_ms: in the end of each speech chunk wait for min_silence_duration_ms before separating it.
    """
    save_path = out_json.with_suffix("") if save_slices else None
    try:
        timestamps = list(
            SileroVAD().get_speech_timestamps(
                in_path,
                save_path=save_path,
                flat_layout=False,
                return_seconds=True,
                speech_pad_ms=speech_pad_ms,
                min_silence_duration_ms=min_silence_duration_ms,
            )
        )
        with open(out_json, "w", encoding="utf-8") as fout:
            json.dump(
                {"wav": str(Path(in_path).resolve()), "timestamps": timestamps},
                fout,
                ensure_ascii=False,
            )
    except Exception as e:
        with open(Path(out_json).with_suffix(".log"), "w+") as fout:
            fout.write(f"{in_path}\n{format_exception(e)}\n")


def denoise(in_path, save_path, overwrite=False):
    """
    Denoise an audio file.

    Args:
        in_path: input json file.
        save_path: save path of audio file.
        overwrite: overwrite outputs.
    """
    try:
        df_model, df_state, _ = init_df(f"{df_repo_dir}/DeepFilterNet3")

        data = json.load(open(in_path, encoding="utf-8"))
        progress_bar = tqdm(
            total=len(data["timestamps"]),
            unit="segments",
            bar_format="{l_bar}{bar}{r_bar} | {percentage:.2f}%",
        )

        audio, _ = load_audio(data["wav"], sr=df_state.sr())
        for idx, timestamp in enumerate(data["timestamps"]):
            progress_bar.update(1)
            slice_path = f"{save_path}/{idx:05d}.wav"
            if os.path.exists(slice_path) and not overwrite:
                continue
            start = int(timestamp["start"] * df_state.sr())
            end = int(timestamp["end"] * df_state.sr())
            enhanced = enhance(df_model, df_state, audio[:, start:end])
            save_audio(slice_path, enhanced, df_state.sr())
    except Exception as e:
        with open(Path(save_path).with_suffix(".log"), "w+") as fout:
            fout.write(f"{in_path}\n{format_exception(e)}\n")


def transcribe(in_path, out_json):
    """
    Transcribe audio files using Paraformer.

    Args:
        in_wav: input audio file.
        out_json: output json file.
    """
    try:
        asr_model = AutoModel(model="paraformer-zh")
        punct_model = AutoModel(model="ct-punc")
        lre_pipeline = pipeline(
            task="speech-language-recognition",
            model="iic/speech_campplus_five_lre_16k",
        )
        panns_model = AudioTagging(
            checkpoint_path=f"{panns_repo_dir}/Cnn14_mAP=0.431.pth"
        )
        pyannote_model = PyannoteONNX()

        data = json.load(open(in_path, encoding="utf-8"))
        long_audio_16k, _ = librosa.load(data["wav"], sr=16000)
        long_audio_32k, _ = librosa.load(data["wav"], sr=32000)
        space_patten = re.compile(r"([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])")
        for idx, timestamp in enumerate(data["timestamps"]):
            start = int(timestamp["start"] * 16000)
            end = int(timestamp["end"] * 16000)
            audio_16k = long_audio_16k[start:end]
            audio_32k = long_audio_32k[start * 2 : end * 2]

            text = asr_model.generate(audio_16k)[0].get("text", "").strip()
            text = space_patten.sub(r"\1\2", text)
            text = space_patten.sub(r"\1\2", text)
            if text != "":
                text = punct_model.generate(text)[0].get("text", "").strip()
            data["timestamps"][idx]["text"] = text
            data["timestamps"][idx]["language"] = lre_pipeline([audio_16k]).get("text", [])[0]
            data["timestamps"][idx]["num_speakers"] = pyannote_model.get_num_speakers(
                audio_16k
            )
            tags = {}
            # panns requires audio length to be larger than 9920 samples (310ms)
            if len(audio_32k) >= 9920:
                clipwise_output, _ = panns_model.inference(audio_32k[None, :])
                clipwise_output = clipwise_output[0]
                sorted_indexes = np.argsort(clipwise_output)[::-1]
                # 0. Speech
                # 1. Male speech, man speaking
                # 2. Female speech, woman speaking
                # 3. Child speech, kid speaking
                # 4. Conversation
                # 5. Narration, monologue
                # 7 - 526: Non speech
                for k in range(5):
                    index = sorted_indexes[k]
                    tags[panns_labels[index]] = round(float(clipwise_output[index]), 2)
            data["timestamps"][idx]["tags"] = tags
        with open(out_json, "w", encoding="utf-8") as fout:
            json.dump(data, fout, ensure_ascii=False)
    except Exception as e:
        with open(Path(out_json).with_suffix(".log"), "w+") as fout:
            fout.write(f"{in_path}\n{format_exception(e)}\n")


def process_audios(in_paths, out_paths, processor, overwrite, num_workers):
    """
    Process audio files.

    Args:
        in_paths: input audio files.
        out_paths: output text files.
        processor: processor function.
        overwrite: overwrite outputs.
        num_workers: number of workers to use.
    """
    args = []
    for in_path, out_path in zip(in_paths, out_paths):
        if overwrite or not out_path.is_file() or not out_path.exists():
            args.append([in_path, out_path])
    if len(args) > 0:
        func = processor.func if isinstance(processor, partial) else processor
        suffix = func.__name__
        thread_map(processor, *zip(*args), max_workers=num_workers, desc=suffix)
