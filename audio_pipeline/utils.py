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

import os
import re
import shutil
from functools import partial
from pathlib import Path

import ffmpeg
import pyloudnorm as pyln
import soundfile as sf
from audio_separator.separator import Separator
from loguru import logger
from pydub import AudioSegment
from pyrnnoise import RNNoise
from silero_vad import SileroVAD
from tqdm.contrib.concurrent import thread_map


AUDIO = "aac|alac|flac|m4a|m4b|m4p|mp3|ogg|opus|wav|wma"
VIDEO = "avi|flv|m4v|mkv|mov|mp4|mpeg|mpg|wmv"

separator = Separator()
vad_model = SileroVAD()


def listdir(path, extensions="|".join([AUDIO, VIDEO]), recursive=False, sort=True):
    """
    List files in a directory.

    Args:
        path: directory path.
        extensions: file extensions.
        recursive: search recursively.
        sort: sort files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")
    files = (
        [
            Path(os.path.join(root, filename))
            for root, _, filenames in os.walk(path, followlinks=True)
            for filename in filenames
            if Path(os.path.join(root, filename)).is_file()
        ]
        if recursive
        else [file for file in path.glob("*") if file.is_file()]
    )

    regex = rf"\.({extensions})$"
    files = [
        file
        for file in files
        if not file.name.startswith(".") and re.search(regex, file.name, re.IGNORECASE)
    ]
    if sort:
        files = sorted(files)
    return files


def mkdir(path, clean=False):
    """
    Make a directory.

    Args:
        path: directory path.
        clean: clean directory if exists.
    """
    path = Path(path)
    if path.exists():
        if clean:
            logger.info(f"Cleaning output directory: {path}")
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Output directory already exists: {path}")


def extract(in_path, out_path, sampling_rate, stream=0, channel=0):
    """
    Extract a mono audio from a multi-channel audio or video file.

    Args:
        in_path: A multi-channel audio or video file.
        out_path: The output mono audio file.
        sampling_rate: Sampling rate of the outputs.
        stream: Stream index of the audio to extract. Defaults to 0.
        channel: Channel index of the stream to extract. Defaults to 0.
    """
    # https://trac.ffmpeg.org/wiki/audio%20types
    # https://trac.ffmpeg.org/wiki/AudioChannelManipulation
    (
        ffmpeg.input(in_path)[f"a:{stream}"]
        .filter("pan", **{"mono|c0": f"c{channel}"})
        .filter("aresample", sampling_rate)
        .output(str(out_path), loglevel="quiet", stats=None)
        .overwrite_output()
        .run()
    )


def to_mono(in_path, out_path):
    """
    Extract a mono audio from a multi-channel audio file.

    Args:
        in_path: The input stereo audio file.
        out_path: The output mono audio file.
    """
    (AudioSegment.from_wav(in_path).set_channels(1).export(out_path, format="wav"))


def denoise(in_path, out_path):
    """
    Denoise an audio file.

    Args:
        in_path: input audio file.
        out_path: output audio file.
    Returns:
        speech_probs: speech probabilities
    """
    denoiser = RNNoise(sf.info(in_path).samplerate)
    return list(denoiser.process_wav(in_path, out_path))


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
    audio, sample_rate = sf.read(in_path)
    # peak normalize audio to [peak] dB
    audio = pyln.normalize.peak(audio, peak)
    # measure the loudness first (BS.1770 meter)
    loudness = pyln.Meter(sample_rate, block_size=block_size).integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, loudness)
    sf.write(out_path, audio, sample_rate)


def vad(in_path, save_path, speech_pad_ms=30, min_silence_duration_ms=100):
    """
    Perform voice activity detection (VAD) on an audio file.

    Args:
        in_path: input audio file.
        save_path: output directory.
        speech_pad_ms: final speech chunks are padded by speech_pad_ms each side.
        min_silence_duration_ms: in the end of each speech chunk wait for min_silence_duration_ms before separating it.
    """
    vad_model.reset_states()
    timestamps = vad_model.get_speech_timestamps(
        in_path,
        save_path=save_path,
        flat_layout=False,
        speech_pad_ms=speech_pad_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    return list(timestamps)


def process_audios(in_paths, out_paths, processor, overwrite, num_workers, suffix=None):
    """
    Process audio files concurrently.

    Args:
        in_paths: input audio files.
        output_paths: output audio files.
        processor: processor function.
        suffix: suffix of output file name.
        overwrite: overwrite outputs.
        num_workers: number of workers to use.
    """
    if suffix is None:
        suffix = (
            f".{processor.func.__name__}"
            if isinstance(processor, partial)
            else f".{processor.__name__}"
        )

    args = []
    _out_paths = []
    for in_path, out_path in zip(in_paths, out_paths):
        out_path = out_path.with_stem(f"{in_path.stem}{suffix}")
        _out_paths.append(out_path)
        if not out_path.exists() or overwrite:
            args.append([in_path, out_path])
    if len(args) > 0:
        thread_map(processor, *zip(*args), max_workers=num_workers)
    return _out_paths


def separate_audios(in_paths, out_paths, overwrite):
    """
    Separate audio files to mono vocals.

    Args:
        in_paths: input audio files.
        output_paths: output audio files.
        overwrite: overwrite outputs.
    """
    if separator.model_instance is None:
        separator.load_model()
        # separator.load_model(model_filename="Kim_Vocal_1.onnx")
    model_name = separator.model_instance.model_name

    _out_paths = []
    for in_path, out_path in zip(in_paths, out_paths):
        # change the output_dir of the separator
        separator.model_instance.output_dir = out_path.parent
        out_path = out_path.with_stem(f"{in_path.stem}_(Vocals)_{model_name}")
        # convert the stereo vocals to mono vocals
        mono_vocals = out_path.with_stem(f"{in_path.stem}.vocals")
        _out_paths.append(mono_vocals)
        if not out_path.exists() or overwrite:
            assert out_path.name in separator.separate(in_path)
            to_mono(out_path, mono_vocals)
    return _out_paths
