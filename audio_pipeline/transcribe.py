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
import soundfile as sf
import whisperx
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from silero_vad import SileroVAD
from tqdm.contrib.concurrent import thread_map

from utils.file import list_files, make_dirs


def trans(in_wav, out_txt, model):
    audio = whisperx.load_audio(in_wav)
    text = model.transcribe(audio, batch_size=16)[0]["text"]
    with open(out_txt, "w") as fout:
        fout.write(f"{in_wav}\t{text}\n")


def trans_zh(in_wav, out_txt, inference_pipeline):
    text = inference_pipeline(in_wav)[0]["text"]
    with open(out_txt, "w") as fout:
        fout.write(f"{in_wav}\t{text}\n")


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite outputs")
@click.option("--clean/--no-clean", default=False, help="Clean outputs before")
@click.option("--num-workers", default=1, help="Number of workers to use")
def main(
    input_dir: str, output_dir: str, recursive: bool, overwrite: bool, clean: bool, num_workers: int
):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)
    files = list_files(input_dir, recursive=recursive)
    logger.info(f"Found {len(files)} files, processing...")

    # model = whisperx.load_model("large-v3", "cuda", compute_type="float16")
    # fn = partial(trans, model=model)
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        model_revision="v2.0.4",
    )
    fn = partial(trans_zh, inference_pipeline=inference_pipeline)

    skipped = 0
    args = []
    for fin in files:
        fout = output_dir / fin.relative_to(input_dir).with_suffix(".txt")
        if not fout.parent.exists():
            fout.parent.mkdir(parents=True)
        if fout.exists() and not overwrite:
            skipped += 1
            continue
        args.append([str(fin), str(fout)])

    if len(files) > 0:
        thread_map(fn, *zip(*args), max_workers=num_workers)

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
