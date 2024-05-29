#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Set up Environment."""

from iopath.common.file_io import PathManagerFactory
import utils.distributed as du
import os


_ENV_SETUP_DONE = False
pathmgr = PathManagerFactory.get(key="pyslowfast")
checkpoint_pathmgr = PathManagerFactory.get(key="pyslowfast_checkpoint")


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

def make_dir(path):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    # Create the checkpoint dir from the master process
    if du.is_master_proc() and not pathmgr.exists(path):
        try:
            pathmgr.mkdirs(path)
        except Exception:
            pass
    return path
