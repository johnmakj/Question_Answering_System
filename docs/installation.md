# Installation

## Operating System (OS)

The project is developed, tested and documented for a Linux-based distribution. Specifically, the project is being
developed under the `Ubuntu` GNU/Linux 20.04.2 LTS x86_64\`\` operating system. If a non-Linux installation is needed,
one has to suitably adapt some of the actions below.

## System-Wide Dependencies

The following system-wide dependencies are needed in order to have a complete development environment:


1. [git](https://git-scm.com/) (>=v2.30.0)


2. [Python](https://www.python.org/) (>=3.10.12, <3.11)

Also, in order to ensure proper connectivity between all components, please install these additional dependencies
below, via `apt`:

```bash
$ apt install python3-pip  # Python package installer
$ apt install python3-tk  # Tkinter - Writing Tk applications with Python 3.x
$ apt install python3-dev  # header files and a static library for Python
```

## Getting started

Clone the project at a local workspace of your choosing with:

```bash
$ cd /where/you/want/the/project/to/exist
$ git clone git@github.com:johnmakj/Question_Answering_System.git
```

Then, at the root of the newly created folder:

```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install poetry
```

This will install the project’s dependency manager, [poetry]([https://python-poetry.org](https://python-poetry.org)).

To install all the project dependencies (required and dev) that will be needed during development, simply type:

```bash
$ poetry install
```

**NOTE**: An error `NoCompatiblePythonVersionFound` may arise if your system-wide python is not compatible with the
specification requested in the `pyproject.toml`.

In that case, use your package installer (`apt`, `packman`, etc.) to explicitly install `python3.8` in your
system. After ensuring that this version is included in the PATH, the command `poetry env use python3.10` will
now ensure that the installation can proceed without problems.

For every other “poetry install” related error (e.g. `TooManyRedirects`), something like:

```bash
$ poetry cache clear pypi --all
```

and re-install should probably solve it.

If nothing works, create a virtual environment for python 3.10 with pyenv and pip install the packages mentioned in the 
pyproject.toml file.