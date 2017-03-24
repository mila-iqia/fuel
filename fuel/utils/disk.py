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
        base = len(force_perm_path) - len(intermediary_folders)

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