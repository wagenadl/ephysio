#!/bin/sh

python3 -m build

twine upload dist/ephysio-1.0.12-py3-none-any.whl


