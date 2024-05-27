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

from functools import partial
from pathlib import Path

import click
from loguru import logger

import utils


@click.command()
@click.argument("in_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option("--stream", default=0, help="Stream index of the audio to extract")
@click.option("--channel", default=0, help="Channel index of the stream to extract")
@click.option("--sampling-rate", default=None, help="Sampling rate of the outputs")
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite outputs")
@click.option("--clean/--no-clean", default=False, help="Clean outputs before")
@click.option("--num-workers", default=4, help="Number of workers to use")
@click.option("--extract/--no-extract", default=True, help="Extract audio with ffmpeg")
@click.option("--loudnorm/--no-loudnorm", default=True, help="Loudnorm audios")
@click.option("--vad/--no-vad", default=True, help="Slice long audios into slices")
@click.option(
    "--separate/--no-separate", default=False, help="Separate vocals of slices"
)
@click.option("--denoise/--no-denoise", default=False, help="Denoise audio slices")
def main(
    in_dir,
    out_dir,
    stream,
    channel,
    sampling_rate,
    recursive,
    overwrite,
    clean,
    num_workers,
    extract,
    loudnorm,
    vad,
    separate,
    denoise,
):
    if extract:
        wavs = f"{out_dir}/wavs"
        in_paths, out_paths = utils.generate_paths(in_dir, wavs, recursive, clean)
        extract = partial(
            utils.extract, stream=stream, channel=channel, sampling_rate=sampling_rate
        )
        utils.process_audios(in_paths, out_paths, extract, overwrite, num_workers)
        in_dir = wavs

    if loudnorm:
        normed = f"{out_dir}/normed"
        in_paths, out_paths = utils.generate_paths(in_dir, normed, recursive, clean)
        utils.process_audios(
            in_paths, out_paths, utils.loudnorm, overwrite, num_workers
        )
        in_dir = normed

    if vad:
        slices = f"{out_dir}/slices"
        in_paths, out_paths = utils.generate_paths(in_dir, slices, recursive, clean)
        save_paths = [path.with_suffix("") for path in out_paths]
        vad = partial(utils.vad, speech_pad_ms=30, min_silence_duration_ms=100)
        utils.process_audios(in_paths, save_paths, vad, overwrite, num_workers)
        in_dir = slices

    if separate:
        vocals = f"{out_dir}/vocals"
        in_paths, out_paths = utils.generate_paths(in_dir, vocals, recursive, clean)
        utils.separate_audios(in_paths, out_paths, overwrite)
        in_dir = vocals

    if denoise:
        denoised = f"{out_dir}/denoised"
        in_paths, out_paths = utils.generate_paths(in_dir, denoised, recursive, clean)
        utils.process_audios(in_paths, out_paths, utils.denoise, overwrite, num_workers)
        in_dir = denoised
    logger.info("Done!")


if __name__ == "__main__":
    main()
