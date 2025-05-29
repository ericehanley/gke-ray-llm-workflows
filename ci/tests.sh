#!/bin/bash

# don't use nbcovert or jupytext unless you're willing
# to check each subprocess unit and validate that errors
# aren't being consumed/hidden
python ci/nb2py.py README.ipynb README.py  # convert notebook to script
python README.py  # run generated script
rm README.py  # remove the generated script
