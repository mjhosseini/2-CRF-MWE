'''
Created on Sep 29, 2012

@author: Nathan Schneider (nschneid)
'''
from __future__ import absolute_import, print_function
import timeit, sys

import pyximport; pyximport.install()
import discriminativeTagger

import os
import sys

@timeit.Timer
def go():
    try:
        print (sys.argv)
        discriminativeTagger.main(sys.argv[1:])
        #os.system('python discriminativeTagger %s'%sys.argv)
    except KeyboardInterrupt:
        raise

print(go.timeit(number=1), file=sys.stderr)
