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
  download $dataset
done

cd -
