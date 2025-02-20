#!/bin/bash
### clean ALL the tests in the subdirectories
OWD=$PWD
SCRIPT_FOLDER=$(realpath $(dirname $0))
cd $SCRIPT_FOLDER
    for i in */
    do
        cd $i
        ./clean.sh
        cd ..
    done
cd $OWD
