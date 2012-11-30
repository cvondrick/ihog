import os
#os.environ['DISTUTILS_DEBUG'] = "1"
from distutils.core import setup, Extension
import distutils.util
import numpy

# includes numpy : package numpy.distutils , numpy.get_include()
# python setup.py build --inplace
# python setup.py install --prefix=dist, 
incs = ['.'] + map(lambda x: os.path.join('spams',x),[ 'linalg', 'prox', 'decomp', 'dictLearn']) + [numpy.get_include()]

osname = distutils.util.get_platform()
cc_flags = ['-fPIC', '-fopenmp']
link_flags = ['-fopenmp', '-s' ]
libs = ['stdc++', 'blas', 'lapack' ]
libdirs = []

if osname.startswith("macosx"):
    cc_flags = ['-fPIC', '-fopenmp','-m32']
    link_flags = ['-m32', '-framework', 'Python']

if osname.startswith("win32"):
    cc_flags = ['-fPIC', '-fopenmp','-DWIN32']
    link_flags = ['-fopenmp', '-mwindows']
    path = os.environ['PATH']
    os.environ['PATH'] = 'C:/MinGW/bin;' + path
    libs = ['stdc++', 'Rblas', 'Rlapack' ]
    libdirs = ['C:/Program Files/R/R-2.15.1/bin/i386']

if osname.startswith("win-amd64"):
    cc_flags = ['-openmp', '-DWIN32', '-DCYGWIN', '-DWINDOWS', '-I','C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/include']
    link_flags = []
    libs = [ 'Rblas', 'Rlapack' ]
    libdirs = ['C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/lib/amd64','C:/Program Files/R/R-2.15.1/bin/x64']


##path = os.environ['PATH']; print "XX OS %s, path %s" %(osname,path)

spams_wrap = Extension(
    '_spams_wrap',
    sources = ['spams_wrap.cpp'],
    include_dirs = incs,
    extra_compile_args = ['-DNDEBUG', '-DUSE_BLAS_LIB'] + cc_flags,
    library_dirs = libdirs,
    libraries = libs,
    # strip the .so
    extra_link_args = link_flags,
    language = 'c++',
    depends = ['spams.h'],
)

def mkhtml(d = None,base = 'sphinx'):
    if d == None:
        d = base
    else:
        d = os.path.join(base,d)
    if not os.path.isdir(base):
        return []
    hdir = d

    l1 = os.listdir(hdir)
    l = []
    for s in l1:
        s = os.path.join(d,s)
        if not os.path.isdir(s):
            l.append(s)
    return l


setup (name = 'spams',
       version= '2.3',
       description='Python interface for SPAMS',
       author = 'Julien Mairal',
       author_email = 'nomail',
       url = 'http://',
       ext_modules = [spams_wrap,],
       py_modules = ['spams', 'spams_wrap', 'myscipy_rand'],
#       scripts = ['test_spams.py'],
       data_files = [
        ('test',['test_spams.py', 'test_decomp.py', 'test_dictLearn.py', 'test_linalg.py', 'test_prox.py', 'test_utils.py']),
        ('doc',['doc_spams.pdf', 'python-interface.pdf']), 
        ('doc/sphinx/_sources',mkhtml('_sources')),
        ('doc/sphinx/_static',mkhtml('_static')),
        ('doc/sphinx',mkhtml()),
        ('doc/html',mkhtml(base = 'html')),
        ('extdata',['boat.png', 'lena.png'])
        ],
)
