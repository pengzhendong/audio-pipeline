import os
import re
import shutil
from pathlib import Path
from typing import Union

from loguru import logger


AUDIO = "aac|alac|flac|m4a|m4b|m4p|mp3|ogg|opus|wav|wma"
VIDEO = "avi|flv|m4v|mkv|mov|mp4|mpeg|mpg|wmv"


def list_files(
    path: Union[Path, str],
    extensions: str = "|".join([AUDIO, VIDEO]),
    recursive: bool = False,
    sort: bool = True,
) -> list[Path]:
    """List files in a directory.
    Args:
        path (Path): Path to the directory.
        extensions (str, optional): Extensions to filter.
        recursive (bool, optional): Whether to search recursively. Defaults to False.
        sort (bool, optional): Whether to sort the files. Defaults to True.
    Returns:
        list: List of files.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")

    files = (
        [
            Path(os.path.join(root, filename))
            for root, _, filenames in os.walk(path, followlinks=True)
            for filename in filenames
            if Path(os.path.join(root, filename)).is_file()
        ]
        if recursive
        else [f for f in path.glob("*") if f.is_file()]
    )
    # skip hidden files
    files = [f for f in files if not f.name.startswith(".")]
    if extensions:
        ext_regex = rf"\.({extensions})$"
        files = [f for f in files if re.search(ext_regex, str(f), re.IGNORECASE)]
    if sort:
        files = sorted(files)
    return files


def make_dirs(path: Union[Path, str], clean: bool = False):
    """Make directories.
    Args:
        path (Union[Path, str]): Path to the directory.
        clean (bool, optional): Whether to clean the directory. Defaults to False.
    """
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        if clean:
            logger.info(f"Cleaning output directory: {path}")
            shutil.rmtree(path)
        else:
            logger.info(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
