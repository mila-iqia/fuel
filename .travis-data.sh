#!/bin/bash
# Download and convert given datasets if not present already
# Usage: .travis-download.sh [dataset ...]
set -ev

function download {
  if [ ! -f $FUEL_DATA_PATH/$1.hdf5 ]; then
    fuel-download $@
    fuel-convert $@
    fuel-download $@ --clear
  fi
}

cd $FUEL_DATA_PATH

for dataset in "$@"; do
  if [ "$dataset" == "ilsvrc2010" ]; then
    wget "http://www.image-net.org/challenges/LSVRC/2010/download/ILSVRC2010_devkit-1.0.tar.gz"
    wget "http://www.image-net.org/challenges/LSVRC/2010/ILSVRC2010_test_ground_truth.txt"
  else
    download $dataset
  fi
done



cd -
