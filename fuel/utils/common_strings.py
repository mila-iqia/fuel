environment_variable_essay = """
Platform-specific instructions for setting environment variables:
Linux
=====
On most linux setups, you can define your environment variable by adding this
line to your ~/.bashrc file:
export PYLEARN2_VIEWER_COMMAND="eog --new-instance"
*** YOU MUST INCLUDE THE WORD "export". DO NOT JUST ASSIGN TO THE ENVIRONMENT VARIABLE ***
If you do not include the word "export", the environment variable will be set
in your bash shell, but will not be visible to processes that you launch from
it, like the python interpreter.
Don't forget that changes from your .bashrc file won't apply until you run
source ~/.bashrc
or open a new terminal window. If you're seeing this from an ipython notebook
you'll need to restart the ipython notebook, or maybe modify os.environ from
an ipython cell.
Mac OS X
========
Environment variables on Mac OS X work the same as in Linux, except you should
modify and run the "source" command on ~/.profile rather than ~/.bashrc.
"""

viewer_command_error_essay = """
PYLEARN2_VIEWER_COMMAND not defined. PLEASE READ THE FOLLOWING MESSAGE
CAREFULLY TO SET UP THIS ENVIRONMENT VARIABLE:
pylearn2 uses an external program to display images. Because different
systems have different image programs available, pylearn2 requires the
 user to specify what image viewer program to use.

You need to choose an image viewer program that pylearn2 should use.
Then tell pylearn2 to use that image viewer program by defining your
PYLEARN2_VIEWER_COMMAND environment variable.

You need to choose PYLEARN_VIEWER_COMMAND such that running

${PYLEARN2_VIEWER_COMMAND} image.png

in a command prompt on your machine will do the following:
    -open an image viewer in a new process.
    -not return until you have closed the image.

Platform-specific recommendations follow.

Linux
=====

Acceptable commands include:
    gwenview
    eog --new-instance

This is assuming that you have gwenview or a version of eog that supports
--new-instance installed on your machine. If you don't, install one of those,
or figure out a command that has the above properties that is available from
your setup.

Mac OS X
========

Acceptable commands include:
    open -Wn

"""
