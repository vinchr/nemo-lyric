#!/usr/bin/env bash

#clone git repo for jamendo...
git clone https://github.com/f90/jamendolyrics

#apply the patch
cd jamendolyrics
pwd
patch < ../Evaluate.sample.patch

#execute evaulation
python Evaluate.py
cd ..
