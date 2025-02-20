SCRIPT_FOLDER=$(realpath $(dirname $0))
cd $SCRIPT_FOLDER
    rm -r run __pycache__ .pytest_cache molsys.log 2>/dev/null
cd - >/dev/null
