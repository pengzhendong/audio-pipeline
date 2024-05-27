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
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from funasr import AutoModel
from loguru import logger
from modelscope.pipelines import pipeline
from panns_inference import AudioTagging, labels
from pyannote_onnx import PyannoteONNX
from pydub import AudioSegment
from pyrnnoise import RNNoise
from silero_vad import SileroVAD
from tqdm.contrib.concurrent import thread_map


AUDIO = "aac|alac|flac|m4a|m4b|m4p|mp3|mp4a|ogg|opus|wav|wma"
VIDEO = "avi|flv|m4v|mkv|mov|mp4|mpeg|mpg|wmv"

lre_pipeline = None
converter = None
normalizer = None
panns_model = None
pyannote_model = None
separator = None
vad_model = SileroVAD()
panns_labels = []
for label in labels:
    label = label.lower().split(", ")[0].replace(" ", "_")
    panns_labels.append(label)


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


def generate_paths(in_dir, out_dir, recursive=False, clean=False, suffix=".wav"):
    """
    Generate input and output paths.

    Args:
        in_dir: input directory.
        out_dir: output directory.
        recursive: search recursively.
        clean: clean output directory if exists.
        suffix: suffix of the output files.
    """
    in_paths = []
    out_paths = []
    if in_dir == out_dir and clean:
        logger.error("You are trying to clean the input directory, aborting.")
        return in_paths, out_paths

    files = listdir(in_dir, recursive=recursive)
    for in_path in files:
        out_path = Path(out_dir) / in_path.relative_to(in_dir).with_suffix(suffix)
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True)
        in_paths.append(in_path)
        out_paths.append(out_path)
    return in_paths, out_paths


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
        error_log = out_path.with_suffix(".log")
        with open(error_log, "w") as fout:
            fout.write(f"{in_path}\n{e}\n")


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
        error_log = out_path.with_suffix(".log")
        with open(error_log, "w") as fout:
            fout.write(f"{in_path}\n{e}\n")


def vad(in_path, save_path, speech_pad_ms=30, min_silence_duration_ms=100):
    """
    Perform voice activity detection (VAD) on an audio file.

    Args:
        in_path: input audio file.
        save_path: output directory.
        speech_pad_ms: final speech chunks are padded by speech_pad_ms each side.
        min_silence_duration_ms: in the end of each speech chunk wait for min_silence_duration_ms before separating it.
    """
    try:
        done = Path(os.path.join(save_path, "done"))
        if done.exists():
            return []
        vad_model.reset_states()
        timestamps = list(
            vad_model.get_speech_timestamps(
                in_path,
                save_path=save_path,
                flat_layout=False,
                speech_pad_ms=speech_pad_ms,
                min_silence_duration_ms=min_silence_duration_ms,
            )
        )
        if done.parent.exists():
            done.touch()
        else:
            logger.warning(f"The output result of vad is empty: {in_path}")
        return timestamps
    except Exception as e:
        error_log = save_path.with_suffix(".log")
        with open(error_log, "w") as fout:
            fout.write(f"{in_path}\n{e}\n")


def denoise(in_path, out_path):
    """
    Denoise an audio file.

    Args:
        in_path: input audio file.
        out_path: output audio file.
    Returns:
        speech_probs: speech probabilities
    """
    try:
        info = sf.info(in_path)
        denoiser = RNNoise(info.samplerate, info.channels)
        return list(denoiser.process_wav(in_path, out_path))
    except Exception as e:
        error_log = out_path.with_suffix(".log")
        with open(error_log, "w") as fout:
            fout.write(f"{in_path}\n{e}\n")


def to_mono(in_path, out_path):
    """
    Extract a mono audio from a multi-channel audio file.

    Args:
        in_path: The input stereo audio file.
        out_path: The output mono audio file.
    """
    AudioSegment.from_wav(in_path).set_channels(1).export(out_path, format="wav")


def separate_audios(in_paths, out_paths, overwrite):
    """
    Separate audio files to mono vocals.

    Args:
        in_paths: input audio files.
        output_paths: output audio files.
        overwrite: overwrite outputs.
    """
    global separator
    if separator is None:
        from audio_separator.separator import Separator

        separator = Separator()
        separator.load_model()
        # separator.load_model(model_filename="Kim_Vocal_1.onnx")
    model_name = separator.model_instance.model_name

    for in_path, out_path in zip(in_paths, out_paths):
        # change the output_dir of the separator
        separator.model_instance.output_dir = out_path.parent
        instrumental = out_path.with_stem(f"{in_path.stem}_(Instrumental)_{model_name}")
        vocals = out_path.with_stem(f"{in_path.stem}_(Vocals)_{model_name}")
        if not out_path.exists() or overwrite:
            try:
                paths = separator.separate(in_path)
                assert vocals.name == paths[0]
                assert instrumental.name == paths[1]
                os.remove(instrumental)
                # convert the stereo vocals to mono
                to_mono(vocals, out_path)
                os.remove(vocals)
            except Exception as e:
                error_log = out_path.with_suffix(".log")
                with open(error_log, "w") as fout:
                    fout.write(f"{in_path}\n{e}\n")


def paraformer_transcribe(in_wav, out_txt, model, batch_size):
    """
    Transcribe audio files using Paraformer.

    Args:
        in_wav: input audio file.
        out_txt: output text file.
        model: funasr auto model.
        batch_size: batch size for decoding.
    """
    try:
        segments = model.generate(str(in_wav), batch_size_s=batch_size)
        text = ""
        for segment in segments:
            text += segment["text"]
        with open(out_txt, "w", encoding="utf-8") as fout:
            fout.write(f"{in_wav}\t{text}\n")
    except Exception as e:
        error_log = out_txt.with_suffix(".log")
        with open(error_log, "w") as fout:
            fout.write(f"{in_wav}\n{e}\n")


def whisper_transcribe(in_wav, out_txt, model, language, batch_size):
    """
    Transcribe audio files using Whisper.

    Args:
        in_wav: input audio file.
        out_txt: output text file.
        model: whisper model.
        language: language spoken in the audio.
        batch_size: batch size for decoding.
    """
    try:
        audio = whisperx.load_audio(in_wav)
        segments = model.transcribe(audio, language=language, batch_size=batch_size)[
            "segments"
        ]
        text = ""
        for segment in segments:
            text += segment["text"]
        with open(out_txt, "w", encoding="utf-8") as fout:
            if language == "zh":
                text = converter.convert(text)
                text = normalizer.normalize(text)
            fout.write(f"{in_wav}\t{text}\n")
    except Exception as e:
        error_log = out_txt.with_suffix(".log")
        with open(error_log, "w") as fout:
            fout.write(f"{in_wav}\n{e}\n")


def language_detection(in_wav, out_txt):
    try:
        global lre_pipeline
        if lre_pipeline is None:
            lre_pipeline = pipeline(
                task="speech-language-recognition",
                model="iic/speech_campplus_five_lre_16k",
            )
        audio, _ = librosa.load(in_wav, sr=16000)
        language = lre_pipeline([audio])["text"][0]
        with open(out_txt, "w", encoding="utf-8") as fout:
            fout.write(f"{in_wav}\t{language}\n")
    except Exception as e:
        error_log = out_txt.with_suffix(".log")
        with open(error_log, "w") as fout:
            fout.write(f"{in_wav}\nDetect language failed.\n")


def panns_tags(in_wav, out_txt):
    try:
        audio, sample_rate = sf.read(in_wav)
        clipwise_output, _ = panns_model.inference(audio[None, :])
        clipwise_output = clipwise_output[0]
        sorted_indexes = np.argsort(clipwise_output)[::-1]
        with open(out_txt, "w", encoding="utf-8") as fout:
            tags = ""
            labels = np.array(panns_labels)
            # 0. Speech
            # 1. Male speech, man speaking
            # 2. Female speech, woman speaking
            # 3. Child speech, kid speaking
            # 4. Conversation
            # 5. Narration, monologue
            # 7 - 526: Non speech
            for k in range(3):
                index = sorted_indexes[k]
                tags += f" {panns_labels[index]}:{clipwise_output[index]:.2f}"
            fout.write(f"{in_wav}\t{tags.strip()}\n")
    except Exception as e:
        error_log = out_txt.with_suffix(".log")
        with open(error_log, "w") as fout:
            fout.write(f"{in_wav}\n{e}\n")


def pyannote_speakers(in_wav, out_txt):
    try:
        num_speakers = pyannote_model.get_num_speakers(in_wav)
        with open(out_txt, "w", encoding="utf-8") as fout:
            fout.write(f"{in_wav}\t{num_speakers}\n")
    except:
        error_log = out_txt.with_suffix(".log")
        with open(error_log, "w") as fout:
            fout.write(f"{in_wav}\nGet num of speakers failed.\n")


def process_audios(
    in_paths, out_paths, processor, overwrite, num_workers, batch_size=1
):
    """
    Process audio files using Paraformer or Whisper.

    Args:
        in_paths: input audio files.
        output_paths: output text files.
        processor: processor function.
        overwrite: overwrite outputs.
        num_workers: number of workers to use.
        batch_size: batch size for transcribing.
    """
    func = processor.func if isinstance(processor, partial) else processor
    suffix = func.__name__.split("_")[0]

    global converter
    global normalizer
    global panns_model
    global pyannote_model
    if suffix == "paraformer":
        model = AutoModel(
            model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc-c"
        )
        processor = partial(processor, model=model, batch_size=batch_size)
    elif suffix == "whisper":
        if converter is None:
            from pyopenhc import OpenHC

            converter = OpenHC("t2s")
        if normalizer is None:
            from tn.chinese.normalizer import Normalizer

            normalizer = Normalizer()
        import whisperx

        model = whisperx.load_model("large-v3", "cuda", compute_type="float16")
        processor = partial(
            whisper_transcribe, model=model, batch_size=batch_size, language="zh"
        )
    elif suffix == "panns" and panns_model is None:
        panns_model = AudioTagging()
        # panns_model = AudioTagging("panns_data/Cnn14_mAP=0.431.pth")
    elif suffix == "pyannote" and pyannote_model is None:
        pyannote_model = PyannoteONNX()

    args = []
    for in_path, out_path in zip(in_paths, out_paths):
        if overwrite or not out_path.is_file() or not out_path.exists():
            args.append([in_path, out_path])
    if len(args) > 0:
        thread_map(processor, *zip(*args), max_workers=num_workers, desc=suffix)
