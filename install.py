# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import launch
from packaging import version
from modules import paths_internal


def install():
    try:
        if not launch.is_installed("onnxruntime-directml") or version.parse("onnxruntime-directml") <= version.parse("1.16.2"):
            launch.run_pip('install onnxruntime-directml>=1.16.1', "onnxruntime-directml")
    except Exception as e:
            print(e)
            print('Warning: Failed to install onnxruntime-directml package, DirectML extension will not work.')

    DML_UNET_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-dml")
    if not os.path.exists(DML_UNET_MODEL_DIR):
        os.makedirs(DML_UNET_MODEL_DIR)

install()
