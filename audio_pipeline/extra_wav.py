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

from pathlib import Path

import click
import ffmpeg
from loguru import logger
from tqdm import tqdm

from utils.file import list_files, make_dirs


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--sampling-rate", default="44100", help="Sampling rate of output files")
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--overwrite/--no-overwrite", default=False, help="Overwrite existing files"
)
@click.option(
    "--clean/--no-clean", default=False, help="Clean output directory before processing"
)
def main(
    input_dir: str,
    output_dir: str,
    sampling_rate: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)
    files = list_files(input_dir, recursive=recursive)
    logger.info(f"Found {len(files)} files, converting to wav")

    skipped = 0
    for file in tqdm(files):
        # Get relative path to input_dir
        relative_path = file.relative_to(input_dir)
        new_file = (
            output_dir
            / relative_path.parent
            / relative_path.name.replace(file.suffix, ".wav")
        )

        if new_file.parent.exists() is False:
            new_file.parent.mkdir(parents=True)
        if new_file.exists() and not overwrite:
            skipped += 1
            continue
        # https://wiki.tnonline.net/w/Blog/Audio_normalization_with_FFmpeg
        ffmpeg.input(str(file)).audio.filter("loudnorm", I=-23).output(
            str(new_file), ac="1", ar=sampling_rate
        ).run(quiet=True)

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
