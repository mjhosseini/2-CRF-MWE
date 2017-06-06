import numpy as np

from scidbpy import connect, SciDBQueryError, SciDBArray

sdb = connect('http://localhost:8080')