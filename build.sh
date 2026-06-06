#!/usr/bin/env bash
set -e
apt-get install -y ffmpeg
pip install --upgrade pip
pip install --only-binary=:all: numpy scipy
pip install -r requirements.txt
