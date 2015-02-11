# installing cc_routines

#from distutils.core import setup
#from Cython.Build import cythonize
#import numpy


#setup(name = "cc_routines",
#      ext_modules = cythonize("./cc_routines.pyx", include_dirs = [numpy.get_include()]))
#setup(name = "cc_routines",
#      ext_modules = cythonize("./cc_routines.pyx", include_dirs = [numpy.get_include()]))
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line

#os.environ["CC"] = "g++" 
#os.environ["CXX"] = "g++"

ext_modules = [Extension("cc_routines", ["cc_routines.pyx"])]

setup(
  name = 'CCD',
    cmdclass = {'build_ext': build_ext},
      include_dirs = [np.get_include()],         # <---- New line
        ext_modules = ext_modules
        )
