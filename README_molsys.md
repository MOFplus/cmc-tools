# molsys

Molsys is the library used to read, write and manipulate atomistic models.
It can be used in python programs and features also lots of scripts to run on the command line to transform structure files.

### Installing

In order to install molsys, clone this repository into the destination of your choosing (we always use /home/%USER/sandbox/, also the installation instructions below use this path)

```
https://github.com/MOFplus/molsys.git
```
or if you have put an ssh key to github
```
git@github.com:MOFplus/molsys.git
```

Afterwards the PATH and PYTHONOATH have to be updated. Add to your .bashrc :
```
export PYTHONPATH=/home/$USER/sandbox/molsys:$PYTHONPATH
export PATH=/home/$USER/sandbox/molsys/scripts:$PATH
```

Mandatory dependencies are:

* numpy (pip install numpy)
* 

Some addons and utility modules require additional packages to be installed. These are:

* [graph-tool](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions#installation-via-package-managers) (molsys.addon.graph) (ppa via apt-get)
* spglib (molsys.addon.spg) (pip)
* pandas (molsys.addon.zmat) (pip)
* scipy (molsys.addon.zmat) (pip)
* [ff_gen](https://github.com/MOFplus/ff_gen) (molsys.addon.ric) (github)

## Running the tests

There will soon be a testing framework framework available.
Currently, there are few tests with an inconsistent way to run them. For instance:
- `lqg_test.py` needs `python lqg_test.py $net_name` and only 5 of them are available (you need to check the code). 
- `toper` and `acab` tests run with just `pytest`.

Same story for examples:
- the only main example (w/o considering addon) is making HKUST-1 and needs `weaver`.

A design policy is TBA.

## Building the Documentation

Mandatory dependencies to built the documentation can be obtained via pip:
```
pip install Sphinx
pip install sphinx-rtd-theme
```
The Documentation can be compiled by running
```
make html
```
in the doc folder.
A Built directory containing
```
/built/html/index.html
```
was created. It can be opened with the browser of your choice

## Contributing

* Any changes to the main mol class (mol.py) have to be assured by [Rochus Schmid](https://github.com/rochusschmid)
* Use google style docstrings ([here](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) in depth):
```python
def new_function(param1, param2)
	"""
	This is a new function.

	Args:
		param1: This is the first param.
		param2: This is a second param.

	Returns:
		This is a description of what is returned.

	Raises:
		KeyError: Raises an exception.
	"""
```

## License

TBA

## Acknowledgments

TBA
