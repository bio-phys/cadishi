#!/bin/sh
PYCACHE="__pycache__"
if [ -e $PYCACHE ]
then
    rm -rf $PYCACHE
fi
py.test
