# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import sys

from .__about__ import __version__ as __version__


def _add_project_root_to_sys_path() -> None:
    """Ensure the Video World repo root is importable for top-level `methods.*` modules."""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "methods").is_dir() and (parent / "models").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


def _check_cuda_extra():
    """Check if CUDA extra is installed."""
    try:
        import cosmos_cuda
    except ImportError:
        raise RuntimeError("CUDA extra not installed. Please run 'uv sync --extra=<cuda_name>'") from None

    if __version__ != cosmos_cuda.__version__:
        raise RuntimeError(
            f"CUDA extra version mismatch: {cosmos_cuda.__version__} != {__version__}. Please run 'uv sync --extra=<cuda_name>'"
        )


_add_project_root_to_sys_path()
_check_cuda_extra()
