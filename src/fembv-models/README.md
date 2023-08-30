FEM-BV models
=============

Implementation of FEM-BV models. To install from source, run

   python -m pip install .

It is recommended that the package be installed into a custom
environment. For example, to install into a custom conda
environment, first create the environment via

    conda create -n fembv-models-env python=3.8
    conda activate fembv-models-env

Note that on MacOS it may be necessary to first install alternative
BLAS libraries or install scs (a dependency of cvxpy) manually beforehand
using pip or conda.

The package may then be installed using

    cd /path/to/package/directory
    python -m pip install .

Optionally, a set of unit tests may be run by executing

    pytest

Alternatively, if tox is installed, it may be used to run the tests using

    tox -c tox.ini
