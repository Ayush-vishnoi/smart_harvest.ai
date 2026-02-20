#!/usr/bin/env python
"""
Bootstrap script that patches numpy before loading the main app
"""
import sys
import os

# Fix numpy compatibility issue FIRST, before any other imports
import numpy as np
from numpy.random.mtrand import RandomState
np.random._mt19937 = RandomState
np.random.mt19937 = RandomState

# Now exec the main app
os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
with open('backend/app_v2.py') as f:
    code = f.read()
    # Remove the bootstrap code from the loaded file to avoid duplication
    lines = code.split('\n')
    # Find the first line that's not a comment or blank after our bootstrap
    exec('\n'.join(lines[30:]))  # Skip the first ~30 lines (bootstrap)

