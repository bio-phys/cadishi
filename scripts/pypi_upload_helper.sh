#!/bin/bash

# run from the project's base directory

python setup.py clean

python3 setup.py sdist

echo "NOW RUN:  twine upload dist/*"

