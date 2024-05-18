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
@click.argument("wav_scp", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--model",
    type=click.Choice(["paraformer", "whisper"]),
    default="paraformer",
    help="ASR model",
)
@click.option(
    "--language",
    type=click.Choice(["en", "zh"]),
    default="zh",
    help="ASR language",
)
@click.option("--asr/--no-asr", default=True, help="Do ASR")
@click.option("--batch_size", default=16, help="Batch size for ASR")
@click.option("--panns/--no-panns", default=False, help="Get audio tags")
@click.option("--pyannote/--no-pyannote", default=False, help="Get num of speakers")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite outputs")
@click.option("--num-workers", default=1, help="Number of workers to use")
def main(
    wav_scp, model, language, asr, panns, pyannote, overwrite, num_workers, batch_size
):
    if asr:
        if model == "paraformer":
            processor = utils.paraformer_transcribe
        elif model == "whisper":
            processor = utils.whisper_transcribe
        utils.transcribe_audios(wav_scp, processor, overwrite, num_workers, batch_size)
    if panns:
        utils.transcribe_audios(wav_scp, utils.panns_tags, overwrite, num_workers)
    if pyannote:
        utils.transcribe_audios(
            wav_scp, utils.pyannote_speakers, overwrite, num_workers
        )
    logger.info("Done!")


if __name__ == "__main__":
    main()
