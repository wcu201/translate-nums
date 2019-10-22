#!/bin/bash

python3 -m venv translate
source translate/bin/activate
pip install -r requirements.txt
python3 main.py

