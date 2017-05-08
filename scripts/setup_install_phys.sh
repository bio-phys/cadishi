#!/bin/bash

# create modulefile, load it, then run this script

set -e

if [[ -z "${CAPRIQORN_HOME}" ]]; then
    echo "(create and) load a capriqorn module first"
else

echo "Ready to install cadishi/capriqorn into ${CAPRIQORN_HOME} ?"
echo "If unsure press ctrl-c now!"
read
mkdir -p ${CAPRIQORN_HOME}/lib/python2.7/site-packages
python setup.py clean config build install --prefix=${CAPRIQORN_HOME}

fi

