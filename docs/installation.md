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
$ sudo apt update
$ sudo apt-get update
$ sudo apt upgrade -y
```

## Getting started

Clone the project at a local workspace of your choosing with:

```bash
$ cd /where/you/want/the/project/to/exist
$ git clone git@github.com:johnmakj/question_answering_system.git
```

Then, at the root of the newly created folder:

```bash
$ python3 -m pip install --upgrade pip
$ pip3 install -r requirements.txt
$ python3 -m spacy download en_core_web_sm
$ export PYTHONPATH=/path/to/the/project:$PYTHONPATH
```

This will install the project’s dependencies and set up the system to run the scripts.
