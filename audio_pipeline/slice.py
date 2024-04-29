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
from loguru import logger
from silero_vad import SileroVAD
from tqdm import tqdm
import soundfile as sf
from tqdm.contrib.concurrent import thread_map

from utils.file import list_files, make_dirs


def vad(in_wav, save_path):
    model = SileroVAD()
    timestamps = model.get_speech_timestamps(
        in_wav,
        save_path=save_path,
        flat_layout=False,
        speech_pad_ms=100,
        min_silence_duration_ms=300
    )
    return list(timestamps)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--clean/--no-clean", default=False, help="Clean outputs before")
@click.option("--num-workers", default=8, help="Number of workers to use")
def main(
    input_dir: str, output_dir: str, recursive: bool, clean: bool, num_workers: int
):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)
    files = list_files(input_dir, recursive=recursive)
    logger.info(f"Found {len(files)} files, processing...")

    args = []
    for fin in files:
        save_path = output_dir / fin.relative_to(input_dir).with_suffix("")
        args.append([str(fin), str(save_path)])
    if len(files) > 0:
        thread_map(vad, *zip(*args), max_workers=num_workers)

    logger.info("Done!")
    logger.info(f"Total: {len(files)}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
