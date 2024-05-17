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
@click.option("--sampling-rate", default="44100", help="Sampling rate of the outputs")
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite outputs")
@click.option("--clean/--no-clean", default=False, help="Clean outputs before")
@click.option("--num-workers", default=1, help="Number of workers to use")
@click.option("--extract/--no-extract", default=True, help="Extract audio with ffmpeg")
@click.option("--separate/--no-separate", default=False, help="Separate vocals")
@click.option("--denoise/--no-denoise", default=False, help="Denoise audios")
@click.option("--loudnorm/--no-loudnorm", default=True, help="Loudnorm audios")
@click.option("--vad/--no-vad", default=True, help="Slice long audios into slices")
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
    separate,
    denoise,
    loudnorm,
    vad,
):
    if in_dir == out_dir and clean:
        logger.error("You are trying to clean the input directory, aborting.")
        return
    utils.mkdir(out_dir, clean)
    files = utils.listdir(in_dir, recursive=recursive)
    logger.info(f"Found {len(files)} files.")

    # prepare input paths and output paths
    in_paths = []
    out_paths = []
    for in_path in files:
        out_path = Path(out_dir) / in_path.relative_to(in_dir)
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True)
        in_paths.append(in_path)
        out_paths.append(out_path)

    if extract:
        extract = partial(
            utils.extract, sampling_rate=sampling_rate, stream=stream, channel=channel
        )
        in_paths = utils.process_audios(
            in_paths, out_paths, extract, overwrite, num_workers, suffix=""
        )
    if separate:
        in_paths = utils.separate_audios(in_paths, out_paths, overwrite)
    if denoise:
        in_paths = utils.process_audios(
            in_paths, out_paths, utils.denoise, overwrite, num_workers
        )
    if loudnorm:
        in_paths = utils.process_audios(
            in_paths, out_paths, utils.loudnorm, overwrite, num_workers
        )
    if vad:
        vad = partial(utils.vad, speech_pad_ms=30, min_silence_duration_ms=100)
        save_paths = [path.with_suffix("") for path in out_paths]
        utils.process_audios(
            in_paths, save_paths, vad, overwrite, num_workers, suffix=""
        )
        with open(f"{out_dir}/wav.scp", "w", encoding="utf-8") as fout:
            for in_path, out_path in zip(in_paths, out_paths):
                for file in utils.listdir(out_path.parent / in_path.stem):
                    fout.write(f"{file.resolve()}\n")

    logger.info("Done!")
    logger.info(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
