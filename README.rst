.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://raw.githubusercontent.com/m-dml/MARINADE/main/interrogate.svg

.. image:: https://raw.githubusercontent.com/m-dml/MARINADE/main/pipeline.svg

====================================
Documentation of MARINADE
====================================

MARINADE is a set of differentiable re-implementations of diffusion and advection operators from GOTM
and marine biogeochemical models inspired by FABM.

Full documentation will follow soon

--------------------------------------------

Installation
------------
To install you have to clone the repository and install it with pip:

.. code-block:: bash

   git clone https://github.com/m-dml/MARINADE
   cd MARINADE
   pip install -e .

--------------------------------------------

Building this documentation
---------------------------
To build this documentation, you have to install the requirements in the `docs_environment.yaml` or
`dev_environment.yaml`. Install `requirements.txt` and MARINADE using `pip`. Then from within the `docs`
folder, you can build the documentation with:

.. code-block:: bash

   make html

The created documentation can be opened in a browser using the `docs/build/html/index.html` file.


--------------------------------------------


Contributing
------------

1. Create an issue describing what you want to change/add/remove ...
2. Create a pull request from that issue. With each pull request, a branch is automatically created.
3. On your machine switch to that newly created branch of that MR.

    .. code-block:: bash

        git switch branch-name

4. Implement your changes. You can push as many commits as you want to this branch.
5. As soon as you are done with all changes, make sure that all existing unit-tests pass, and if you implemented something new, make sure it also got a unit-test
6. Run pre-commit (see below)
7. Push one more time if the pre-commit run changed your code.
8. Mark your PR as ready and tell a maintainer that your PR is ready for being reviewed.
9. If the reviewer is happy, they will merge your code to the main branch. Else you might have to make a few adjustments.

Pre-Commit
----------

Please install the conda environment in `dev_environment.yaml`. Before you commit, please run

.. code-block:: bash

    pre-commit run --all-files


from the main directory to make sure that your code is formatted correctly.
We use `black` for formatting and `flake8` for linting.

Technical Info
--------------
We use a custom backward method for the tridiagonal solve (see documentation).
