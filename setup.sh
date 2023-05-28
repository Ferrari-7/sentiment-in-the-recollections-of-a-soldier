#!/usr/bin/env bash

#create virtual environment
python3 -m venv env
# activate the virtual environment
source ./env/bin/activate

# install hdbscan for BERTopic
sudo apt-get update
sudo apt-get install python3-dev

# requirements
pip install --upgrade pip
pip install --upgrade nbformat
python3 -m pip install -r requirements.txt

# turn the virtual environment "off"
deactivate