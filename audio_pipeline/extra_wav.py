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
import ffmpeg
from loguru import logger
from tqdm.contrib.concurrent import thread_map

from utils.file import list_files, make_dirs


def extract(
    input: str,
    output: str,
    sampling_rate: str,
    stream: int = 0,
    channel: int = 0,
):
    # https://trac.ffmpeg.org/wiki/audio%20types
    # https://trac.ffmpeg.org/wiki/AudioChannelManipulation
    (
        ffmpeg.input(input)[f"a:{stream}"]
        .filter("pan", **{"mono|c0": f"c{channel}"})
        .filter("aresample", sampling_rate)
        .output(output, loglevel="quiet", stats=None)
        .overwrite_output()
        .run()
    )


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--stream", default=0, help="Stream index of the audio to extract")
@click.option("--channel", default=0, help="Channel index of the stream to extract")
@click.option("--sampling-rate", default="44100", help="Sampling rate of the outputs")
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite outputs")
@click.option("--clean/--no-clean", default=False, help="Clean outputs before")
@click.option("--num-workers", default=8, help="Number of workers to use")
def main(
    input_dir: str,
    output_dir: str,
    stream: int,
    channel: int,
    sampling_rate: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    num_workers: int,
):
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

    fn = partial(extract, sampling_rate=sampling_rate, stream=stream, channel=channel)
    if len(args) > 0:
        thread_map(fn, *zip(*args), max_workers=num_workers)

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
