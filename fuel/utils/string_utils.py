""" Utilities for modifying strings"""

import os
import re

from theano.compat.six import string_types
from theano.compat.six.moves import xrange

from fuel.utils.exc import EnvironmentVariableError, NoDataPathError
from fuel.utils.exc import reraise_as
from fuel.utils.common_strings import environment_variable_essay


def preprocess(string, environ=None):
    """
    Preprocesses a string, by replacing `${VARNAME}` with
    `os.environ['VARNAME']` and ~ with the path to the user's
    home directory
    Parameters
    ----------
    string : str
        String object to preprocess
    environ : dict, optional
        If supplied, preferentially accept values from
        this dictionary as well as `os.environ`. That is,
        if a key appears in both, this dictionary takes
        precedence.
    Returns
    -------
    rval : str
        The preprocessed string
    """
    if environ is None:
        environ = {}

    split = string.split('${')

    rval = [split[0]]

    for candidate in split[1:]:
        subsplit = candidate.split('}')

        if len(subsplit) < 2:
            raise ValueError('Open ${ not followed by } before '
                             'end of string or next ${ in "' + string + '"')

        varname = subsplit[0]
        try:
            val = (environ[varname] if varname in environ
                   else os.environ[varname])
        except KeyError:
            if varname == 'PYLEARN2_DATA_PATH':
                reraise_as(NoDataPathError())
            if varname == 'PYLEARN2_VIEWER_COMMAND':
                reraise_as(EnvironmentVariableError(
                    viewer_command_error_essay + environment_variable_essay)
                )

            reraise_as(ValueError('Unrecognized environment variable "' +
                                  varname + '". Did you mean ' +
                                  match(varname, os.environ.keys()) + '?'))

        rval.append(val)

        rval.append('}'.join(subsplit[1:]))

    rval = ''.join(rval)

    string = os.path.expanduser(string)

    return rval



