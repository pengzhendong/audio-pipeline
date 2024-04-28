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
import pyloudnorm as pyln
import soundfile as sf
from tqdm.contrib.concurrent import thread_map

from utils.file import list_files, make_dirs


def loudness_norm(in_path, out_path, peak=-1.0, loudness=-23.0, block_size=0.400):
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        in_wav: input audio file
        out_wav: output audio file
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


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite outputs")
@click.option("--clean/--no-clean", default=False, help="Clean outputs before")
@click.option("--num-workers", default=8, help="Number of workers to use")
@click.option("--peak", default=-1.0, help="Peak norm audio to -1dB")
@click.option("--loudness", help="Loudness norm audio to -23dB LUFS", default=-23.0)
@click.option("--block-size", help="Block size in second for measurement", default=0.4)
def main(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    peak: float,
    loudness: float,
    block_size: float,
    num_workers: int,
):
    """Perform loudness normalization (ITU-R BS.1770-4) on audio files."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)
    files = list_files(input_dir, recursive=recursive)
    logger.info(f"Found {len(files)} files, extracting specified channel to mono wav")

    skipped = 0
    args = []
    for fin in files:
        fout = output_dir / fin.relative_to(input_dir).with_suffix(".wav")
        if not fout.parent.exists():
            fout.parent.mkdir(parents=True)
        if fout.exists() and not overwrite:
            skipped += 1
            continue
        args.append([str(fin), str(fout)])

    fn = partial(loudness_norm, peak=peak, loudness=loudness, block_size=block_size)
    if len(args) > 0:
        thread_map(fn, *zip(*args), max_workers=num_workers)

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
