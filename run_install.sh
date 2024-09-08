#!/bin/bash
echo "Install python library"
python -m pip install -U pip
pip install -r requirement.txt
python -m ipykernel install --user --name venv --display-name "mlbase"
pip install -e .
echo "completed"