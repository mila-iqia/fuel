# -*- coding: utf-8 -*-

# Some of code below is taken from
# [pylearn2](https://github.com/lisa-lab/pylearn2) framework developed under
# the copyright:
#
# Copyright (c) 2011--2014, Université de Montréal
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Filesystem utility code."""
import os


def disk_usage(path):
    """Return free usage about the given path, in bytes.

    Parameters
    ----------
    path : str
        Folder for which to return disk usage

    Returns
    -------
    output : tuple
        Tuple containing total space in the folder and currently
        used space in the folder

    """
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    return total, used


def safe_mkdir(folder_name, force_perm=None):
    """Create the specified folder.

    If the parent folders do not exist, they are also created.
    If the folder already exists, nothing is done.

    Parameters
    ----------
    folder_name : str
        Name of the folder to create.
    force_perm : str
        Mode to use for folder creation.

    """
    if os.path.exists(folder_name):
        return
    intermediary_folders = folder_name.split(os.path.sep)

    # Remove invalid elements from intermediary_folders
    if intermediary_folders[-1] == "":
        intermediary_folders = intermediary_folders[:-1]
    if force_perm:
        force_perm_path = folder_name.split(os.path.sep)
        if force_perm_path[-1] == "":
            force_perm_path = force_perm_path[:-1]

    for i in range(1, len(intermediary_folders)):
        folder_to_create = os.path.sep.join(intermediary_folders[:i + 1])

        if os.path.exists(folder_to_create):
            continue
        os.mkdir(folder_to_create)
        if force_perm:
            os.chmod(folder_to_create, force_perm)


def check_enough_space(dataset_local_dir, remote_fname, local_fname,
                       max_disk_usage=0.9):
    """Check if the given local folder has enough space.

    Check if the given local folder has enough space to store
    the specified remote file.

    Parameters
    ----------
    remote_fname : str
        Path to the remote file
    remote_fname : str
        Path to the local folder
    max_disk_usage : float
        Fraction indicating how much of the total space in the
        local folder can be used before the local cache must stop
        adding to it.

    Returns
    -------
    output : boolean
        True if there is enough space to store the remote file.

    """
    storage_need = os.path.getsize(remote_fname)
    storage_total, storage_used = disk_usage(dataset_local_dir)

    # Instead of only looking if there's enough space, we ensure we do not
    # go over max disk usage level to avoid filling the disk/partition
    return ((storage_used + storage_need) <
            (storage_total * max_disk_usage))
