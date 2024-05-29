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

import click
from loguru import logger

import utils


@click.command()
@click.argument("in_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option(
    "--model",
    type=click.Choice(["paraformer", "whisper"]),
    default="paraformer",
    help="ASR model",
)
@click.option("--asr/--no-asr", default=True, help="Do ASR")
@click.option("--batch_size", default=16, help="Batch size for ASR")
@click.option("--detect-lang/--no-detect-lang", default=False, help="Detect language")
@click.option("--panns/--no-panns", default=False, help="Get audio tags")
@click.option("--pyannote/--no-pyannote", default=False, help="Get num of speakers")
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite outputs")
@click.option("--clean/--no-clean", default=False, help="Clean outputs before")
@click.option("--num-workers", default=1, help="Number of workers to use")
def main(
    in_dir,
    out_dir,
    model,
    asr,
    detect_lang,
    panns,
    pyannote,
    recursive,
    overwrite,
    clean,
    num_workers,
    batch_size,
):
    if asr:
        in_paths, out_paths = utils.generate_paths(
            in_dir, f"{out_dir}/asr", recursive, clean, ".paraformer.txt"
        )
        utils.process_audios(
            in_paths,
            out_paths,
            utils.transcribe,
            overwrite,
            num_workers,
            batch_size,
        )
    if detect_lang:
        in_paths, out_paths = utils.generate_paths(
            in_dir, f"{out_dir}/lang", recursive, clean, ".txt"
        )
        utils.process_audios(
            in_paths, out_paths, utils.language_detection, overwrite, num_workers
        )
    if panns:
        in_paths, out_paths = utils.generate_paths(
            in_dir, f"{out_dir}/tags", recursive, clean, ".txt"
        )
        utils.process_audios(
            in_paths, out_paths, utils.panns_tags, overwrite, num_workers
        )
    if pyannote:
        in_paths, out_paths = utils.generate_paths(
            in_dir, f"{out_dir}/speakers", recursive, clean, ".txt"
        )
        utils.process_audios(
            in_paths, out_paths, utils.pyannote_speakers, overwrite, num_workers
        )
    logger.info("Done!")


if __name__ == "__main__":
    main()
