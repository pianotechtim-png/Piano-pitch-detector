#!/usr/bin/env bash
set -e
pip install --upgrade pip
pip install --only-binary=:all: numpy scipy==1.11.4
pip install flask flask-cors librosa==0.10.1 soundfile audioread gunicorn
