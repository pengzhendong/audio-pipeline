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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import librosa
from loguru import logger
from silero_vad import SileroVAD
from tqdm import tqdm

from utils.file import AUDIO_EXTENSIONS, list_files, make_dirs


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--clean/--no-clean", default=True, help="Clean output directory before processing"
)
@click.option(
    "--num-workers",
    help="Number of workers to use for processing, defaults to number of CPU cores",
    default=os.cpu_count(),
    show_default=True,
    type=int,
)
def main(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    clean: bool,
    num_workers: int,
):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)
    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, processing...")

    vad = SileroVAD()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []
        for file in tqdm(files, desc="Preparing tasks"):
            # Get relative path to input_dir
            relative_path = file.relative_to(input_dir)
            save_path = output_dir / relative_path.parent / relative_path.stem
            if save_path.parent.exists() is False:
                save_path.parent.mkdir(parents=True)

            tasks.append(
                executor.submit(
                    vad.get_speech_timestamps,
                    wav_path=str(file),
                    save_path=save_path,
                    flat_layout=True,
                    threshold=0.5,
                    speech_pad_ms=30,
                    min_silence_duration_ms=100,
                    max_speech_duration_s=30,
                    min_speech_duration_ms=250,
                )
            )

        for i in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            assert i.exception() is None, i.exception()

    logger.info("Done!")
    logger.info(f"Total: {len(files)}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
