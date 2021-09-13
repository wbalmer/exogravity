The exoGravity package
======================

Installation
------------

### Requirements
-   Tested on Linux/Fedora 33, 34
-   Python 2.7 or Python 3.x, ideally with x\>=5 (required for
    ruamel.yaml)
-   Astropy (\>=2.0.2)
-   ruamel.yaml \>=0.16.5 (recommended) or alternatively pyyaml \>=3.11
-   NumPy (\>=1.16.5), Scipy (\>=0.19.0)
-   cleanGravity (see installation procedure below)

### Installation procedure

Download or clone cleanGravity: https://gitlab.obspm.fr/mnowak/cleanGravity
Add the directory to your python path, so that you can then import the package
in any of your python code: "import cleanGravity as cg"

Download or clone exoGravity, https://gitlab.obspm.fr/mnowak/exoGravity and make
all the following scripts executable (`chmod +x`): create_config.py, create_phase_reference.py, astrometry_reduce.py, spectrum_reduce.py, swap_reduce.py, normalize_spectrum.py, plot_spectrum.py.
Optionally, you can the directory to your path, so that you can execute these scripts from anywhere.

### Tutorial data

The data directory inside the tutorial (exoGravity/tutorial/data) must be populated with some reduced GRAVITY data.
These can be downloaded using this link (about 1.5GB):
https://share.obspm.fr/s/AinfW9JAPRzDYbF

### Suggested order for the notebooks

If you just want to run the exoGravity reduction, simply look at exogravity.ipynb.

Otherwise, if you also want to understand a bit how it works, and manipulate data:
1- data_manipulation.ipynb
2- swap_example.ipynb
3- onaxis_example.ipynb
4- exogravity.ipynb
