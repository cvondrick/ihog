features.so : features.pyx
	cython -X boundscheck=False,cdivision=True,infer_types=True -o features.cc features.pyx
	g++ -shared -pthread -fPIC -fwrapv -O3 -fno-strict-aliasing `python2.6-config --includes` -o features.so features.cc
