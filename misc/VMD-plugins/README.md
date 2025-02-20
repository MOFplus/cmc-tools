# molsys & mofplus file reader plugins for VMD

This folder contains file reader plugins for the famous [VMD](https://www.ks.uiuc.edu/Research/vmd/) visualiation software. Currently, only the former mfp5 type files are supported to be red. The .mfpx file plugin will be avilable soon

## Installation 

### mfp5 file reader (former mfp5!)

This works only when vmd is installed in its default path /usr/local/lib/vmd. If that's not the case, modifiy the Makefile as needed.

To check if everything works, try:

```
cd mfp5
make all
```
and investigate the output to see if there's any errors. Ignore the warnings ;)  You should have two shared objects: `libmfp5.so` and `mfptrajplugin.so

`
To install the plugin (i.e. to copy the shared objects, try
```
cd mfp5
make install
```


### mfpx file reader

To install the plugin (i.e. to copy the shared objects, try
```
cd mfpx
make install
```



## Usage

### mfp

You can start off vmd with -mfp5 or -mfpx to tell it to read such a file as

```
vmd -mfp5 [FILENAME]
vmd -mfpx [FILENAME]
```

# CREDITS

The mfp5 vmd plugin was inspired by the [h5md vmd plugin](https://github.com/h5md/VMD-h5mdplugin) and would have been much harder to built without the h5md plugin providing the layout for the plugin structure and hdf5 i/o examples. 
