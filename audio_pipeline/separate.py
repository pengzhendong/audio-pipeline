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
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Union

import click
import torch
from demucs import apply, audio, pretrained, separate
from loguru import logger
from tqdm import tqdm

from utils.file import AUDIO, list_files, make_dirs


def init_model(
    name: str = "htdemucs",
    device: Optional[Union[str, torch.device]] = None,
    segment: Optional[int] = None,
) -> torch.nn.Module:
    """
    Initialize the model
    Args:
        name: Name of the model
        device: Device to use
        segment: Set split size of each chunk. This can help save memory of graphic card.
    Returns:
        The model
    """

    model = pretrained.get_model(name)
    model.eval()

    if device:
        model.to(device)
    logger.info(f"Model {name} loaded on {device}")

    if segment:
        if isinstance(model, apply.BagOfModels):
            for m in model.models:
                m.segment = segment
        else:
            model.segment = segment
    return model


def separate_audio(
    model: torch.nn.Module,
    audio_path: Union[Path, str],
    shifts: int = 1,
    num_workers: int = 0,
    progress: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Separate audio into sources
    Args:
        model: The model
        audio_path: The audio path
        shifts: Run the model N times, larger values will increase the quality but also the time
        num_workers: if non zero, device is 'cpu', how many threads to use in parallel
        progress: Show progress bar
    Returns:
        The separated tracks
    """

    audio = separate.load_track(audio_path, model.audio_channels, model.samplerate)
    ref = audio.mean(0)
    audio = (audio - ref.mean()) / audio.std()

    sources = apply.apply_model(
        model,
        audio[None],
        device=next(model.parameters()).device,
        shifts=shifts,
        split=True,
        overlap=0.25,
        progress=progress,
        num_workers=num_workers,
    )[0]
    sources = sources * ref.std() + ref.mean()
    return dict(zip(model.sources, sources))


def merge_tracks(
    tracks: dict[str, torch.Tensor],
    filter: Optional[list[str]] = None,
) -> torch.Tensor:
    """
    Merge tracks into one audio
    Args:
        tracks: The separated audio tracks
        filter: The tracks to merge
    Returns:
        The merged audio
    """

    filter = filter or list(tracks.keys())
    merged = torch.zeros_like(next(iter(tracks.values())))
    for key in tracks:
        if key in filter:
            merged += tracks[key]
    return merged


def worker(
    input_dir: str,
    output_dir: str,
    num_channels: int,
    recursive: bool,
    overwrite: bool,
    track: list[str],
    model: str,
    shifts: int,
    device: torch.device,
    shard_idx: int = -1,
    total_shards: int = 1,
    num_workers: int = 0,
):
    files = list_files(input_dir, extensions=AUDIO, recursive=recursive)
    if shard_idx >= 0:
        files = [f for i, f in enumerate(files) if i % total_shards == shard_idx]
    shard_name = f"[Shard {shard_idx + 1}/{total_shards}]"
    logger.info(f"{shard_name} Found {len(files)} files.")
    if len(files) == 0:
        return

    # TODO: init model outside
    model = init_model(model, device)
    skipped = 0
    for file in tqdm(
        files,
        desc=f"{shard_name} Separating audio",
        position=0 if shard_idx < 0 else shard_idx,
        leave=False,
    ):
        fout = output_dir / file.relative_to(input_dir)
        if not fout.parent.exists():
            fout.parent.mkdir(parents=True)
        if fout.exists() and not overwrite:
            skipped += 1
            continue

        if device.type == "cuda":
            num_workers = 0
        separated = separate_audio(model, file, shifts=shifts, num_workers=num_workers)
        merged = merge_tracks(separated, track)[2 - num_channels :]
        audio.save_audio(merged, fout, model.samplerate)

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--num-channels", default=1, help="Num channels of output files")
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite outputs")
@click.option("--clean/--no-clean", default=False, help="Clean outputs before")
@click.option("--track", "-t", multiple=True, default=["vocals"], help="Target tracks")
@click.option("--model", help="Name of model to use", default="htdemucs")
@click.option("--shifts", default=1, help="Larger shifts will improve quality a bit")
@click.option("--num_workers_per_gpu", default=2, help="Number of workers per GPU")
@click.option("--num-workers", default=8, help="Number of workers to use without GPU")
def main(
    input_dir: str,
    output_dir: str,
    num_channels: int,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    track: list[str],
    model: str,
    shifts: int,
    num_workers_per_gpu: int,
    num_workers: int,
):
    """
    Separates audio in input_dir using model and saves to output_dir.
    """

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)
    fn = partial(
        worker,
        input_dir=input_dir,
        output_dir=output_dir,
        num_channels=num_channels,
        recursive=recursive,
        overwrite=overwrite,
        track=track,
        model=model,
        shifts=shifts,
        num_workers=num_workers,
    )

    gpu_count = torch.cuda.device_count()
    if torch.cuda.is_available() and gpu_count >= 1:
        logger.info(f"Device has {gpu_count} GPUs, let's use them!")
        mp.set_start_method("spawn")

        processes = []
        shards = gpu_count * num_workers_per_gpu
        for shard_idx in range(shards):
            p = mp.Process(
                target=fn,
                kwargs={
                    "device": torch.device(f"cuda:{shard_idx % gpu_count}"),
                    "shard_idx": shard_idx,
                    "total_shards": shards,
                },
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        fn(device=torch.device("cpu"))


if __name__ == "__main__":
    main()
