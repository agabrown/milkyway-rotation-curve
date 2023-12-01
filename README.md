# milkyway-rotation-curve
Milky Way rotation curve from modelling of Gaia proper motions

This repository contains the code to model Gaia proper motions in terms of simple Milky Way rotation curves as desribed in section 10.1 of the paper [Gaia Data Release 3: Golden Sample of Astrophysical Parameters Gaia Collaboration, Creevey, O.L., et al., 2022, A&A](https://doi.org/10.1051/0004-6361/202243800). The repository contents are an adapted/updated version of [this repository](https://github.com/agabrown/milkyway-disk-proper-motions/tree/main/notebooks). In particular the dependencies on  [Gala](http://gala.adrian.pw/en/latest/) were removed and the modelling code was switched to [NumPyro](https://num.pyro.ai/en/stable/).

## Repository contents

* `data` folder with data from Gaia DR3 (data files not stored on Github).
* Jupyter notebooks with code to do the Milky Way rotation curve modelling using [NumPyro](https://num.pyro.ai/en/stable/).
* The old [Stan](https://mc-stan.org/) modelling code (used for the paper).
* `img` folder containing the images used in the notebooks, or produced with the python code in the notebooks (not stored on Github).

## Python dependencies
[NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/), [Astropy](https://www.astropy.org/), [PyGaia](https://pypi.org/project/PyGaia/), [HealPy](https://github.com/healpy/healpy), [ArviZ](https://python.arviz.org/en/latest/), [corner](https://corner.readthedocs.io/en/latest/), [Cartopy](https://scitools.org.uk/cartopy/docs/latest/), [NumPyro](https://num.pyro.ai/en/stable/)

### Dependencies for Stan models (no longer needed)
[CmdStanPy](https://github.com/stan-dev/cmdstanpy)