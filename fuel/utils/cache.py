# -*- coding: utf-8 -*-
#
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
#
"""Dataset preloading tool.

This file provides the ability to make a local cache of a dataset or
part of it. It is meant to help in the case where multiple jobs are
reading the same dataset from ${FUEL_DATA_PATH}, which may cause a
great burden on the network.
With this file, it is possible to make a local copy
(in ${FUEL_LOCAL_DATA_PATH}) of any required file and have multiple
processes use it simultaneously instead of each acquiring its own copy
over the network.
Whenever a folder or a dataset copy is created locally, it is granted
the same access as it has under ${FUEL_LOCAL_DATA_PATH}. This is
guaranteed by default copy.

"""
import logging
import os
import shutil
import stat
import time

from fuel import config
from fuel.utils.disk import safe_mkdir, check_enough_space
from fuel.utils.lock import get_writelock, release_writelock, get_readlock

log = logging.getLogger(__name__)
pid = os.getpid()


def cache_file(filename):
    """Caches a file locally if possible.

    If caching was succesfull, or if
    the file was previously successfully cached, this method returns the
    path to the local copy of the file. If not, it returns the path to
    the original file.

    Parameters
    ----------
    filename : str
        Remote file to cache locally

    Returns
    -------
    output : str
        Updated (if needed) filename to use to access the remote
        file.

    """
    dataset_local_dir = config.local_data_path
    dataset_remote_dir = os.path.dirname(filename)
    if dataset_remote_dir == "" or dataset_local_dir == "":
        log.debug("Local dataset cache is deactivated")

    remote_name = filename

    # Check if a local directory for data has been defined. Otherwise,
    # do not locally copy the data
    if dataset_local_dir == "":
        return filename

    common_msg = ("Message from fuel local cache of dataset"
                  "(specified by the environment variable "
                  "FUEL_LOCAL_DATA_PATH): ")
    # Make sure the file to cache exists and really is a file
    if not os.path.exists(remote_name):
        log.error(
            "Error : Specified file {} does not exist".format(remote_name))
        return filename

    if not os.path.isfile(remote_name):
        log.error(
            "Error : Specified name {} is not a file".format(remote_name))
        return filename

    if not remote_name.startswith(dataset_remote_dir):
        log.warning(
            common_msg +
            "We cache in the local directory only what is"
            " under $FUEL_DATA_PATH: {}".format(remote_name))
        return filename

    # Create the $FUEL_LOCAL_DATA_PATH folder if needed
    safe_mkdir(dataset_local_dir,
               (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
                stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |
                stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH))

    # Determine local path to which the file is to be cached
    local_name = os.path.join(dataset_local_dir,
                              os.path.relpath(remote_name,
                                              dataset_remote_dir))

    # Create the folder structure to receive the remote file
    local_folder = os.path.split(local_name)[0]
    try:
        safe_mkdir(local_folder)
    except Exception:
        log.warning(
            (common_msg +
             "While creating the directory {}, we got an error."
             " We won't cache to the local disk.").format(local_folder))
        return filename

    # Acquire writelock on the local file to prevent the possibility
    # of any other process modifying it while we cache it if needed.
    # Also, if another process is currently caching the same file,
    # it forces the current process to wait for it to be done before
    # using the file.
    if not os.access(local_folder, os.W_OK):
        log.warning(common_msg +
                    "Local folder {} isn't writable."
                    " This is needed for synchronization."
                    " We will use the remote version."
                    " Manually fix the permission."
                    .format(local_folder))
        return filename
    get_writelock(local_name)

    # If the file does not exist locally, consider creating it
    if not os.path.exists(local_name):

        # Check that there is enough space to cache the file
        if not check_enough_space(dataset_local_dir, remote_name,
                                  local_name):
            log.warning(common_msg +
                        "File {} not cached: Not enough free space"
                        .format(remote_name))
            release_writelock()
            return filename

        # There is enough space; make a local copy of the file
        copy_from_server_to_local(dataset_remote_dir, dataset_local_dir,
                                  remote_name, local_name)
        log.info(common_msg + "File {} has been locally cached to {}"
                 .format(remote_name, local_name))
    elif os.path.getmtime(remote_name) > os.path.getmtime(local_name):
        remote_modified_time = time.strftime(
            '%Y-%m-%d %H:%M:%S',
            time.localtime(os.path.getmtime(remote_name)))
        local_modified_time = time.strftime(
            '%Y-%m-%d %H:%M:%S',
            time.localtime(os.path.getmtime(local_name)))
        log.warning(common_msg +
                    "File {} in cache will not be used: The remote file "
                    "(modified {}) is newer than the locally cached file "
                    "{} (modified {})."
                    .format(remote_name, remote_modified_time,
                            local_name, local_modified_time))
        release_writelock()
        return filename
    elif os.path.getsize(local_name) != os.path.getsize(remote_name):
        log.warning(common_msg +
                    "File {} not cached: The remote file ({} bytes) is of "
                    "a different size than the locally cached file {} "
                    "({} bytes). The local cache might be corrupt."
                    .format(remote_name, os.path.getsize(remote_name),
                            local_name, os.path.getsize(local_name)))
        release_writelock()
        return filename
    elif not os.access(local_name, os.R_OK):
        log.warning(common_msg +
                    "File {} in cache isn't readable. We will use the"
                    " remote version. Manually fix the permission."
                    .format(local_name))
        release_writelock()
        return filename
    else:
        log.debug("File {} has previously been locally cached to {}"
                  .format(remote_name, local_name))

    # Obtain a readlock on the downloaded file before releasing the
    # writelock. This is to prevent having a moment where there is no
    # lock on this file which could give the impression that it is
    # unused and therefore safe to delete.
    get_readlock(pid, local_name)
    release_writelock()

    return local_name


def copy_from_server_to_local(dataset_remote_dir, dataset_local_dir,
                              remote_fname, local_fname):
    """Copies a remote file locally.

    Parameters
    ----------
    remote_fname : str
        Remote file to copy
    local_fname : str
        Path and name of the local copy to be made of the remote file.

    """
    log.debug("Copying file `{}` to a local directory `{}`."
              .format(remote_fname, dataset_local_dir))

    head, tail = os.path.split(local_fname)
    head += os.path.sep
    if not os.path.exists(head):
        os.makedirs(os.path.dirname(head))

    shutil.copyfile(remote_fname, local_fname)

    # Copy the original group id and file permission
    st = os.stat(remote_fname)
    os.chmod(local_fname, st.st_mode)
    # If the user have read access to the data, but not a member
    # of the group, he can't set the group. So we must catch the
    # exception. But we still want to do this, for directory where
    # only member of the group can read that data.
    try:
        os.chown(local_fname, -1, st.st_gid)
    except OSError:
        pass

    # Need to give group write permission to the folders
    # For the locking mechanism
    # Try to set the original group as above
    dirs = os.path.dirname(local_fname).replace(dataset_local_dir, '')
    sep = dirs.split(os.path.sep)
    if sep[0] == "":
        sep = sep[1:]
    for i in range(len(sep)):
        orig_p = os.path.join(dataset_remote_dir, *sep[:i + 1])
        new_p = os.path.join(dataset_local_dir, *sep[:i + 1])
        orig_st = os.stat(orig_p)
        new_st = os.stat(new_p)
        if not new_st.st_mode & stat.S_IWGRP:
            os.chmod(new_p, new_st.st_mode | stat.S_IWGRP)
        if orig_st.st_gid != new_st.st_gid:
            try:
                os.chown(new_p, -1, orig_st.st_gid)
            except OSError:
                pass
