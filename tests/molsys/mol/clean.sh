SCRIPT_FOLDER=$(realpath $(dirname $0))
cd $SCRIPT_FOLDER
    rm -r __pycache__ .pytest_cache molsys.log run .hypothesis 2>/dev/null
cd - >/dev/null
