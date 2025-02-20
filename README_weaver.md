# weaver

Weaver can be used to constrict atomistic models of MOFs, COFs, ...  given its Building Blocks and a blueprint (an embedded topology)
Its main advantage is that it automatically detects ambiguities that may arise during the construction process.
for more information visit [MOFplus](https://www.mofplus.org/content/show/rta). 

### Installing

In order to install weaver, clone this repository into the destination of your choosing (we always use /home/%USER/sandbox/, also the installation instructions below use this path)

```
git clone https://github.com/MOFplus/weaver.git
```
or if you have put an ssh key to github
```
git clone git@github.com:MOFplus/weaver.git
```
A small fortran module has to be compiled to a shared library so weaver can use it. Do this via:
```
cd /weaver/frotator
make
```
It will try to compile both, the python2 and the python3 version of the module. 
It is not necessary to have both compiled, check for your python version whether the
proper shared object was made:
* frotator.so (python2)
* frotator3.so (python3)

Afterwards the PATH and PYTHONOATH have to be updated. Add to your .bashrc :
```
export PYTHONPATH=/home/$USER/sandbox/weaver:$PYTHONPATH
export PATH=/home/$USER/sandbox/weaver/scripts:$PATH
```

Mandatory dependencies are:

* numpy (pip install numpy)
* scipy v>=0.17.0 (pip install scipy)
* [molsys](https://github.com/MOFplus/molsys) (github)

Non-critical dependencies are:

* sobol sequences (pip install sobol)


## Running the tests

There will soon be a testing framework framework available.


## Contributing

TBA

## License

TBA

## Acknowledgments

TBA
