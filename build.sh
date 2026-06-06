#!/usr/bin/env bash
set -e
pip install --upgrade pip
pip install --only-binary=:all: numpy scipy
pip install flask flask-cors gunicorn

