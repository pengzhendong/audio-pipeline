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
import re
import shutil
from pathlib import Path

from loguru import logger


AUDIO = "aac|alac|flac|m4a|m4b|m4p|mp3|mp4a|ogg|opus|wav|wma"
VIDEO = "avi|flv|m4v|mkv|mov|mp4|mpeg|mpg|wmv"


def listdir(path, extensions, recursive=False, sort=True):
    """
    List files in a directory.

    Args:
        path: directory path.
        extensions: file extensions.
        recursive: search recursively.
        sort: sort files.
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
        else [file for file in path.glob("*") if file.is_file()]
    )

    regex = rf"\.({extensions})$"
    files = [
        file
        for file in files
        if not file.name.startswith(".") and re.search(regex, file.name, re.IGNORECASE)
    ]
    if sort:
        files = sorted(files)
    return files


def mkdir(path, clean=False):
    """
    Make a directory.

    Args:
        path: directory path.
        clean: clean directory if exists.
    """
    path = Path(path)
    if path.exists():
        if clean:
            logger.info(f"Cleaning output directory: {path}")
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Output directory already exists: {path}")


def generate_paths(
    in_dir,
    out_dir,
    recursive=False,
    clean=False,
    extensions="|".join([AUDIO, VIDEO]),
    output_suffix=".wav",
):
    """
    Generate input and output paths.

    Args:
        in_dir: input directory.
        out_dir: output directory.
        recursive: search recursively.
        clean: clean output directory if exists.
        extensions: file extensions.
        output_suffix: suffix of the output files.
    """
    in_paths = []
    out_paths = []
    if in_dir == out_dir and clean:
        logger.error("You are trying to clean the input directory, aborting.")
        return in_paths, out_paths

    files = listdir(in_dir, extensions=extensions, recursive=recursive)
    for in_path in files:
        out_path = Path(out_dir) / in_path.relative_to(in_dir).with_suffix(
            output_suffix
        )
        if output_suffix == "":
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        in_paths.append(in_path)
        out_paths.append(out_path)
    return in_paths, out_paths


def format_exception(exception):
    file_path = exception.__traceback__.tb_frame.f_code.co_filename
    line_no = exception.__traceback__.tb_lineno
    return f"{exception.__class__.__name__}: {exception} at {file_path}:{line_no}"
