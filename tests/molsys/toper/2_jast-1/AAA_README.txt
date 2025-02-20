# TO RUN THESE TESTS, DO THE FOLLOWING IN THE SAME DIRECTORY:
THIS_README_FILE="AAA_README.txt" #or the position of this file
FOLDER=$(dirname $(realpath $THIS_README_FILE))
cd $FOLDER
cd weave
pytest _test_gen_jast-1.py
cd ..
pytest
### TBI: color assignment is based just on the first atom element!
