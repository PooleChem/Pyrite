# Contributing to Pyrite

Thank you for helping improve Pyrite! Bug reports, questions, ideas and pull requests are all
welcome. For a bug or an idea, open an [issue](https://github.com/PooleChem/Pyrite/issues) first;
for a small fix, a pull request is fine straight away.

## Setting up

Pyrite's dependencies (RDKit, numba, OpenMM and PDBFixer among them) are easiest to install from
conda-forge:

```bash
git clone https://github.com/PooleChem/Pyrite.git
cd Pyrite
conda env create -f doc/environment.yml
conda activate pyrite-docs
pip install -e ".[dev,docs]"  # Pyrite itself, editable, with pytest, ruff and the Sphinx packages
```

The version is in `pyrite/__init__.py` (`__version__`); the package and the documentation read it
from there.

## Checks

Every push and pull request runs the same checks as below on GitHub; a pull request can only be
merged into `dev` or `main` when they pass.

```bash
pytest                                            # the tests, about a minute
ruff check pyrite tests benchmarks                # the code
ruff format pyrite tests benchmarks               # the formatting (fixes it)
make -C doc html                                  # the documentation, runs the user guide
```

The documentation build runs every page of the user guide, and fails when one of them does. The
built site is in `doc/build/html`; after changing a file in `doc/source/_static`, remove
`doc/build/html` first, as Sphinx does not copy changed static files.

## Code

- Formatting and imports follow ruff, with a line length of 100 (see `pyproject.toml`).
- Docstrings follow [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html), with
  Parameters, Returns, See Also and Examples; the documentation build validates them.
- Scoring never changes a `Mol`: positions are computed from a pose, and RDKit work is done on a copy
  (`Mol.to_rdkit`).
- New behaviour comes with tests; a fixed bug with a test that fails without the fix.

## Adding a scoring function

The [Writing a scoring function](https://poolechem.github.io/Pyrite/user_guide/writing_scoring_functions.html)
page of the user guide explains how: a new term needs one method, `_score`. Register it in
`SCORING_FACTORIES` in `tests/conftest.py`, and the conformance tests check it automatically
(scores, batches, gradients, combinations). Grade its speed with `benchmarks/speed_grades.py`.

## Branches and pull requests

- `dev` collects the work in progress; `main` is what is released, on PyPI and in the documentation
  online. Both only change through pull requests whose checks pass.
- Work on a branch of your own (or a fork), and open the pull request against `dev`.
- Keep a pull request to one change, and describe what it changes and why. A maintainer is asked for
  a review automatically, and merges it.

## Releases

1. Branch `release/x.y.z` off `dev`, and set the version in `pyrite/__init__.py`. Only fixes go into
   the release branch.
2. Open a pull request from `release/x.y.z` into `main`; merge it when the checks pass.
3. Tag the merge `vx.y.z` and create a GitHub release from the tag: this publishes the package on
   PyPI, and the documentation of `main` is published on every push.
4. Merge `main` back into `dev`, so that the fixes of the release are not lost.

## License

Pyrite is licensed for non-commercial use (see [LICENSE](LICENSE)). By contributing, you agree that
your contribution is licensed under the same terms.
