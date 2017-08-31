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
"""Utility code to manage filesystem locks."""
import atexit
import logging
import os
import random
import socket  # only used for gethostname()
import time

hostname = socket.gethostname()
logger = logging.getLogger(__name__)
TIMEOUT = 5
MIN_WAIT = 5
NOT_SET = object()


class Unlocker(object):
    """Class wrapper around lock-release mechanism.

    Ensures that the lock is automatically released when the program
    exits (even when crashing or being interrupted), using the __del__
    class method.

    """
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir
        # Keep a pointer to the 'os' module, otherwise it may not be accessible
        # anymore in the __del__ method.
        self.os = os

    def __del__(self):
        """Destructor."""
        self.unlock()

    def unlock(self):
        """Remove current lock.

        This function does not crash if it is unable to properly
        delete the lock file and directory. The reason is that it
        should be allowed for multiple jobs running in parallel to
        unlock the same directory at the same time (e.g. when reaching
        their timeout limit).

        """
        # If any error occurs, we assume this is because someone else tried to
        # unlock this directory at the same time.
        # Note that it is important not to have both remove statements within
        # the same try/except block. The reason is that while the attempt to
        # remove the file may fail (e.g. because for some reason this file does
        # not exist), we still want to try and remove the directory.
        try:
            self.os.remove(self.os.path.join(self.tmp_dir, 'lock'))
        except Exception:
            pass
        try:
            self.os.rmdir(self.tmp_dir)
        except Exception:
            pass


def refresh_lock(lock_file):
    """'Refresh' an existing lock.

    'Refresh' an existing lock by re-writing the file containing the
    owner's unique id, using a new (randomly generated) id, which is also
    returned.

    """
    unique_id = '%s_%s_%s' % (
        os.getpid(),
        ''.join([str(random.randint(0, 9)) for i in range(10)]), hostname)
    try:
        lock_write = open(lock_file, 'w')
        lock_write.write(unique_id + '\n')
        lock_write.close()
    except Exception:
        # In some strange case, this happen.  To prevent all tests
        # from failing, we release the lock, but as there is a
        # problem, we still keep the original exception.
        # This way, only 1 test would fail.
        while get_lock.n_lock > 0:
            release_lock()
        raise
    return unique_id

# NOT_SET is used because None is a valid input for timeout


def lock(tmp_dir, timeout=NOT_SET, min_wait=None, max_wait=None, verbosity=1):
    """Obtain lock.

    Obtain lock access by creating a given temporary directory (whose base
    will be created if needed, but will not be deleted after the lock is
    removed). If access is refused by the same lock owner during more than
    'timeout' seconds, then the current lock is overridden. If timeout is
    None, then no timeout is performed.

    The lock is performed by creating a 'lock' file in 'tmp_dir' that
    contains a unique id identifying the owner of the lock (the process
    id, followed by a random string).

    When there is already a lock, the process sleeps for a random amount
    of time between min_wait and max_wait seconds before trying again.

    If 'verbosity' is >= 1, then a message will be displayed when we need
    to wait for the lock. If it is set to a value >1, then this message
    will be displayed each time we re-check for the presence of the lock.
    Otherwise it is displayed only when we notice the lock's owner has
    changed.

    Parameters
    ----------
    str tmp_dir : str
        Lock directory that will be created when acquiring the lock.

    timeout : int
        Time (in seconds) to wait before replacing an existing lock.

    min_wait : int
        Minimum time (in seconds) to wait before trying again to get the
        lock.

    max_wait : int
        Maximum time (in seconds) to wait before trying again to get the
        lock (default 2 * min_wait).

    verbosity : int
        Amount of feedback displayed to screen (default 1).

    """
    if min_wait is None:
        min_wait = MIN_WAIT
    if max_wait is None:
        max_wait = min_wait * 2
    if timeout is NOT_SET:
        timeout = TIMEOUT
    # Create base of lock directory if required.
    base_lock = os.path.dirname(tmp_dir)
    if not os.path.isdir(base_lock):
        try:
            os.makedirs(base_lock)
        except OSError:
            # Someone else was probably trying to create it at the same time.
            # We wait two seconds just to make sure the following assert does
            # not fail on some NFS systems.
            time.sleep(2)
    assert os.path.isdir(base_lock)

    # Variable initialization.
    lock_file = os.path.join(tmp_dir, 'lock')
    random.seed()
    my_pid = os.getpid()
    no_display = (verbosity == 0)

    nb_error = 0
    # The number of time we sleep when their is no errors.
    # Used to don't display it the first time to display it less frequently.
    # And so don't get as much email about this!
    nb_wait = 0
    # Acquire lock.
    while True:
        try:
            last_owner = 'no_owner'
            time_start = time.time()
            other_dead = False
            while os.path.isdir(tmp_dir):
                try:
                    with open(lock_file) as f:
                        read_owner = f.readlines()[0].strip()

                    # The try is transition code for old locks.
                    # It may be removed when people have upgraded.
                    try:
                        other_host = read_owner.split('_')[2]
                    except IndexError:
                        other_host = ()  # make sure it isn't equal to any host
                    if other_host == hostname:
                        try:
                            # Just check if the other process still exist.
                            os.kill(int(read_owner.split('_')[0]), 0)
                        except OSError:
                            other_dead = True
                        except AttributeError:
                            pass  # os.kill does not exist on windows
                except Exception:
                    read_owner = 'failure'
                if other_dead:
                    if not no_display:
                        msg = "process '%s'" % read_owner.split('_')[0]
                        logger.warning("Overriding existing lock by dead %s "
                                       "(I am process '%s')", msg, my_pid)
                    get_lock.unlocker.unlock()
                    continue
                if last_owner == read_owner:
                    if (timeout is not None and
                            time.time() - time_start >= timeout):
                        # Timeout exceeded or locking process dead.
                        if not no_display:
                            if read_owner == 'failure':
                                msg = 'unknown process'
                            else:
                                msg = "process '%s'" % read_owner.split('_')[0]
                            logger.warning("Overriding existing lock by %s "
                                           "(I am process '%s')", msg, my_pid)
                        get_lock.unlocker.unlock()
                        continue
                else:
                    last_owner = read_owner
                    time_start = time.time()
                    no_display = (verbosity == 0)
                if not no_display and nb_wait > 0:
                    if read_owner == 'failure':
                        msg = 'unknown process'
                    else:
                        msg = "process '%s'" % read_owner.split('_')[0]
                    logger.info("Waiting for existing lock by %s (I am "
                                "process '%s')", msg, my_pid)
                    logger.info("To manually release the lock, delete %s",
                                tmp_dir)
                    if verbosity <= 1:
                        no_display = True
                nb_wait += 1
                time.sleep(random.uniform(min_wait, max_wait))

            try:
                os.mkdir(tmp_dir)
            except OSError:
                # Error while creating the directory: someone else
                # must have tried at the exact same time.
                nb_error += 1
                if nb_error < 10:
                    continue
                else:
                    raise
            # Safety check: the directory should be here.
            assert os.path.isdir(tmp_dir)

            # Write own id into lock file.
            unique_id = refresh_lock(lock_file)

            # Verify we are really the lock owner (this should not be needed,
            # but better be safe than sorry).
            with open(lock_file) as f:
                owner = f.readlines()[0].strip()

            if owner != unique_id:
                # Too bad, try again.
                continue
            else:
                # We got the lock, hoorray!
                return

        except Exception as e:
            # If something wrong happened, we try again.
            logger.warning("Something wrong happened: %s %s", type(e), e)
            nb_error += 1
            if nb_error > 10:
                raise
            time.sleep(random.uniform(min_wait, max_wait))
            continue


def get_lock(lock_dir, **kw):
    """Obtain lock on compilation directory.

    Parameters
    ----------
    lock_dir : str
        Lock directory.
    kw : dict
        Additional arguments to be forwarded to the `lock` function when
        acquiring the lock.

    Notes
    -----
    We can lock only on 1 directory at a time.

    """
    if not hasattr(get_lock, 'n_lock'):
        # Initialization.
        get_lock.n_lock = 0
        if not hasattr(get_lock, 'lock_is_enabled'):
            # Enable lock by default.
            get_lock.lock_is_enabled = True
        get_lock.lock_dir = lock_dir
        get_lock.unlocker = Unlocker(get_lock.lock_dir)
    else:
        if lock_dir != get_lock.lock_dir:
            # Compilation directory has changed.
            # First ensure all old locks were released.
            assert get_lock.n_lock == 0
            # Update members for new compilation directory.
            get_lock.lock_dir = lock_dir
            get_lock.unlocker = Unlocker(get_lock.lock_dir)

    if get_lock.lock_is_enabled:
        # Only really try to acquire the lock if we do not have it already.
        if get_lock.n_lock == 0:
            lock(get_lock.lock_dir, **kw)
            atexit.register(Unlocker.unlock, get_lock.unlocker)
            # Store time at which the lock was set.
            get_lock.start_time = time.time()
        else:
            # Check whether we need to 'refresh' the lock. We do this
            # every 'config.compile.timeout / 2' seconds to ensure
            # no one else tries to override our lock after their
            # 'config.compile.timeout' timeout period.
            if get_lock.start_time is None:
                # This should not happen. So if this happen, clean up
                # the lock state and raise an error.
                while get_lock.n_lock > 0:
                    release_lock()
                raise Exception(
                    "For some unknow reason, the lock was already taken,"
                    " but no start time was registered.")
            now = time.time()
            if now - get_lock.start_time > TIMEOUT:
                lockpath = os.path.join(get_lock.lock_dir, 'lock')
                logger.info('Refreshing lock %s', str(lockpath))
                refresh_lock(lockpath)
                get_lock.start_time = now
    get_lock.n_lock += 1


def release_lock():
    """Release lock on compilation directory."""
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    # Only really release lock once all lock requests have ended.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        get_lock.start_time = None
        get_lock.unlocker.unlock()


def get_writelock(filename):
    """Obtain a writelock on a file.

    Only one write lock may be held at any given time.

    Parameters
    ----------
    filename : str
        Name of the file on which to obtain a writelock

    """
    # write lock expect locks to be on folder. Since we want a lock on a
    # file, we will have to ask write lock for a folder with a different
    # name from the file we want a lock on or else write lock will
    # try to create a folder with the same name as the file
    get_lock(filename + ".writelock")


def release_writelock():
    """Release the previously obtained writelock."""
    release_lock()


def release_readlock(lockdir_name):
    """Release a previously obtained readlock.

    Parameters
    ----------
    lockdir_name : str
        Name of the previously obtained readlock

    """
    # Make sure the lock still exists before deleting it
    if os.path.exists(lockdir_name) and os.path.isdir(lockdir_name):
        os.rmdir(lockdir_name)


def get_readlock(pid, path):
    """Obtain a readlock on a file.

    Parameters
    ----------
    path : str
        Name of the file on which to obtain a readlock

    """
    timestamp = int(time.time() * 1e6)
    lockdir_name = "%s.readlock.%i.%i" % (path, pid, timestamp)
    os.mkdir(lockdir_name)

    # Register function to release the readlock at the end of the script
    atexit.register(release_readlock, lockdir_name=lockdir_name)
