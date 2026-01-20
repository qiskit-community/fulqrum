# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fulqrum helper utilites for input and output"""

import lzma
import orjson
from pathlib import Path

from ..exceptions import FulqrumError


def dict_to_json(dict, filename, overwrite=False):
    """Save dictionary to a JSON or XZ file. File extension can be 'json'
    or 'xz', the latter or which does LZMA compression which is
    recommended for large dictionaries.

    Parameters:
        dict (dict): Dictionary
        filename (str): File to store to
        overwrite (bool): Overwrite file if it exits, default=False
    """
    file = Path(filename)
    if file.is_file() and not overwrite:
        raise Exception("File already exists, set overwrite=True")
    file_type = filename.split(".")[-1].lower()
    if file_type == "json":
        with open(filename, "wb") as fd:
            fd.write(orjson.dumps(dict))
    elif file_type == "xz":
        compressor = lzma.LZMACompressor()
        json_data = orjson.dumps(dict)
        lzma_data = compressor.compress(json_data) + compressor.flush()
        with open(filename, "wb") as fd:
            fd.write(lzma_data)
    else:
        raise FulqrumError("File type must be 'json' or 'xz'")


def json_to_dict(filename):
    """Load dict from a JSON or XZ file.

    Parameters:
        filename (str): File to load from

    Returns:
        dict
    """
    filename = str(filename)  # convert PosixPath
    file_type = filename.split(".")[-1].lower()
    if file_type == "json":
        with open(filename, "r", encoding="utf-8") as fd:
            dic = orjson.loads(fd.read())
    elif file_type == "xz":
        lzma_file = lzma.open(filename)
        data = lzma_file.readlines()
        dic = orjson.loads(data[0])
    else:
        raise FulqrumError("File type must be 'json' or 'xz'")
    return dic
