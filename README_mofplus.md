# mofplus

mofplus is the API library to the [MOFplus](www.mofplus.org) webpage. It can be used to download topologies, bbs, structures, ...

### INSTALLING

In order to install molsys, clone this repository into the destination of your choosing (we always use /home/%USER/sandbox/, also the installation instructions below use this path)

```
https://github.com/MOFplus/mofplus.git or git@github.com:MOFplus/mofplus.git
```

Afterwards the PATH and PYTHONPATH have to be updated. Add to your .bashrc :
```
export PATH=/home/$USER/sandbox/mofplus/scripts:$PATH
export PYTHONPATH=/home/$USER/sandbox/mofplus:$PYTHONPATH
alias mofplus="python3 -i /home/$USER/sandbox/mofplus/mofplus/ff.py"
```


Mandatory dependencies (python3) are:

* numpy (pip3 install numpy)
* registration on [MOFplus](www.mofplus.org) 

Optional dependencies (mandatory for force field assignment) are:
* [molsys](https://github.com/MOFplus/molsys)


## Running the api

Once everything is set up properly, the api can be used simply by typing 
```
mofplus
```
in a terminal. Alternatively, the FF assignment can be used via the `query_parameters` script as
```
query_parameters [MFPXFILENAME] "MOF-FF"
```
where `[MFPXFILENAME]` is the structure to be assigned and `"MOF-FF"` is the forcefield to be assigned 

## Running the tests

There will soon be a testing framework framework available.

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

TBA

## License

TBA

## Acknowledgments


