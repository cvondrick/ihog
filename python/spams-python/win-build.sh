#!/bin/bash
case $1 in -x) set -x;shift;;esac

RBASE="/c/Program Files/R/R-2.15.1"

die () {
    echo $*
    exit 1
}

[ -d "$RBASE" ] || die "No dir $RBASE. You should install R from http://cran.r-project.org/"

x64=0
name=win32
if [ -d /c/Windows/SysWOW64 ]; then
    x64=1
    name=win-amd64
fi

if [ $x64 -ne 0 ]; then
    dstd=build/lib.win-amd64-2.7
    RDIR="$RBASE/bin/x64"
    VC="/c/Program Files (x86)/Microsoft Visual Studio 9.0/VC/bin/amd64"
    PATH="$VC":"/c/Program Files (x86)/Microsoft Visual Studio 9.0/Common7/IDE":$PATH:"$RDIR"
    if [ ! -r  Rblas.lib -o ! -r Rlapack.lib ]; then
	# try to make .lib
	for f in Rblas Rlapack; do
	    pexports "$RDIR"/$f.dll >$f.def 2>/dev/null
	    [ $? -eq 0 ] || die "Sorry! Cannot find pexports.exe to build $f.lib"
	    "$VC"/lib /DEF:$f.def /MACHINE:amd64 /OUT:$f.lib
	done
    fi
    /c/Python27/python setup.py build
    srcd="/c/Windows/winsxs"
    cp -p $srcd/amd64_microsoft.vc90.openmp_*/vcomp90.dll $dstd
    srcd="/c/Program Files (x86)/Microsoft Visual Studio 9.0"/VC/redist/amd64/Microsoft.VC90.CRT/
    cp -p "$srcd"/*.dll $dstd
else
    srcd=/c/MinGW/bin
    dstd=build/lib.win32-2.7
    RDIR="$RBASE/bin/i386"
    /c/Python27/python setup.py build -c mingw32
    cp -p $srcd/libgomp-1.dll $srcd/"libstdc++-6.dll" $srcd/libgcc_s_dw2-1.dll $srcd/pthreadGC2.dll $dstd
fi

cp -p "$RDIR"/*.dll $dstd
/c/Python27/python setup.py bdist_wininst
if [ -r Release-name ]; then
    vers=`cat Release-name`
    f=`/bin/ls dist/*$name*.exe`
    tl=`echo $f | sed "s|^dist/.*\.$name||"`
    mv $f spams-python-$vers.$name$tl
fi
exit 0

