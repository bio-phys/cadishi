#!/bin/bash

# run from the project's base directory

python setup.py clean

/opt/apps/anaconda/3/current/bin/python3 setup.py sdist
#/opt/apps/anaconda/3/4.2.0/bin/python3 setup.py bdist_wheel --universal

echo "NOW RUN:  twine upload dist/*"

