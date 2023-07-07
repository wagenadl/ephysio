#!/bin/sh

python3 -m build
twine upload dist/ephysio-1.0.1.tar.gz

