#!/usr/bin/env bash

# Get the script path
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t 3dteethland_processing "$SCRIPTPATH"