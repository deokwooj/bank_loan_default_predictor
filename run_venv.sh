#!/bin/bash
pyenv local 3.9
venv_dir="./.venv"

# Check for the --test flag
if [[ $1 == "--test" ]]; then
    venv_dir="./venv_test"
    echo "Installing virtual env in $venv_dir"
elif [[ -n $1 ]]; then
    venv_dir="./$1"
    echo "Installing virtual env in $venv_dir"
else
    echo "Installing virtual env in $venv_dir"
fi

# Remove the existing virtual environment
rm -rf "$venv_dir"

echo "Resetting venv"
rm -rf "$venv_dir"

echo "Creating virtual env at $venv_dir"
python -m venv "$venv_dir"

echo "To activate virtual environment, run ..."
echo "source $venv_dir/bin/activate"