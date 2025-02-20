#!/bin/bash
### run ALL the tests in the subdirectories
OWD=$PWD
SCRIPT_FOLDER=$(realpath $(dirname $0))
cd $SCRIPT_FOLDER
    for i in */
    do
        cd $i
        ./clean.sh
        pytest3 -vs --disable-warnings
        cd ..
    done
cd $OWD
