clear all;
get_architecture;

%%%%%%%%%%%%% COMPILER CONFIGURATION %%%%%%%%%%%%%%%%
% set up the compiler you want to use. Possible choices are
%   - 'mex' (default matlab compiler), this is the easy choice if your matlab
%           is correctly configured. Note that this choice might not compatible
%           with the option 'use_multithread=true'. 
%   - 'icc' (intel compiler), usually produces the fastest code, but the
%           compiler is not free and not installed by default.
%   - 'gcc' (gnu compiler), good choice (for Mac, use gcc >= 4.6 for
%           the multi-threaded version, otherwise set use_multithread=false).
%           For windows, you need to have cygwin installed.
%   - 'open64' (amd compiler), optimized for opteron cpus.
%   - 'vs'  (visual studio compiler) for windows computers (10.0 or more is recommended)
%            for some unknown reason, the performance obtained with vs is poor compared to icc/gcc
compiler='mex';

 %%%%%%%%%%%% BLAS/LAPACK CONFIGURATION %%%%%%%%%%%%%%
% set up the blas/lapack library you want to use. Possible choices are
%   - builtin: blas/lapack shipped with Matlab, 
%           same as mex: good choice if matlab is correctly configured.
%   - mkl: (intel math kernel library), usually the fastest, but not free.
%   - acml: (AMD Core math library), optimized for opteron cpus
%   - blas: (netlib at atlas version of blas/lapack), free
% ==> you can also tweak this script to include your favorite blas/lapack library
blas='builtin';

%%%%%%%%%%%% MULTITHREADING CONFIGURATION %%%%%%%%%%%%%%
% set true if you want to use multi-threaded capabilities of the toolbox. You
% need an appropriate compiler for that (intel compiler, most recent gcc, or visual studio pro)
use_multithread=true;   % (might not compatible with compiler=mex)
% if the compilation fails on Mac, try the single-threaded version.
% to run the toolbox on a cluster, it can be a good idea to deactivate this

% if you use the options 'mex' and 'builtin', you can proceed with the compilation by
% typing 'compile' in the matlab shell. Otherwise, you need to set up a few path below.

%%%%%%%%%%%% PATH CONFIGURATION %%%%%%%%%%%%%%%%%%%%
% only if you do not use the options 'mex' and 'builtin'
% set up the path to the compiler libraries that you intend to use below
if strcmp(compiler,'gcc') 
    if linux || mac
       % example when compiler='gcc' for Linux/Mac:   (path containing the files libgcc_s.*)
       path_to_compiler_libraries='/usr/lib/x86_64-linux-gnu/gcc/x86_64-linux-gnu/4.5/';
       path_to_compiler='/usr/bin/';
    else
       % example when compiler='gcc' for Windows+cygwin:   (the script does not
       % work at the moment in this configuration
       path_to_compiler='C:\cygwin\bin\';
       path_to_compiler_libraries='C:\cygwin\lib\gcc\i686-pc-cygwin\4.5.3\';
    end
elseif strcmp(compiler,'open64') 
   % example when compiler='gcc' for Linux/Mac:   (path containing libgcc_s.*)
   path_to_compiler_libraries='/opt/amdsdk/v1.0/x86_open64-4.2.4/lib/gcc-lib/x86_64-open64-linux/4.2.4/';
   path_to_compiler='/opt/amdsdk/v1.0/x86_open64-4.2.4/bin/';
elseif strcmp(compiler,'icc')
    if linux || mac
       % example when compiler='icc' for Linux/Mac
       path_to_compiler_libraries='/opt/intel/composerxe/lib/intel64/';
       path_to_compiler='/opt/intel/composerxe/bin/';
    else
       % example when compiler='icc' for Windows
       path_to_compiler_libraries='C:\Program Files (x86)\Intel\Composer XE\compiler\lib\intel64\';
       path_to_compiler='C:\Program Files (x86)\Intel\Composer XE\bin\intel64\';
       path_to_compiler_include='C:\Program Files (x86)\Intel\Composer XE\compiler\include\';
       path_to_vs='C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\amd64\';
    end
elseif strcmp(compiler,'vs')
    % example when compiler='vs' for Windows
    path_to_compiler_libraries='C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\amd64\';
    path_to_compiler=path_to_compiler_libraries;
elseif strcmp(compiler,'mex')
   % leave it blank when compiler='mex'
   path_to_compiler_libraries='';
   path_to_compiler='';
%   use_multithread=false;
end

% set up the path to the blas/lapack libraries. 
if strcmp(blas,'mkl')
   if linux || mac
      path_to_blas='/opt/intel/composerxe/mkl/lib/intel64/';
   else
      path_to_blas='C:\Program Files (x86)\Intel\Composer XE\mkl\lib\intel64\';
   end
elseif strcmp(blas,'blas')
   if linux || mac
       path_to_blas='/usr/lib/';
   else
       path_to_blas='?';
   end
elseif strcmp(blas,'acml')
   if linux || mac
       %path_to_blas=' /opt/amdsdk/v1.0/acml/open64_64/lib/';
       path_to_blas='/opt/acml5.0.0/ifort64/lib/';
   else
       path_to_blas='?';
   end
elseif strcmp(blas,'builtin')
    % leave it to /usr/lib/ for built-in:
    path_to_blas='/';
end
   
debug=false;
if debug 
   use_multithread=false;
end
%%%%%%%%%%%% END OF THE CONFIGURATION %%%%%%%%%%%%%%
% Do not touch what is below this line, unless you know what you are doing
out_dir='./build/';

COMPILE = { 
            % compile dictLearn toolbox
            '-I./linalg/ -I./decomp/ -I./dictLearn/ dictLearn/mex/mexTrainDL.cpp', 
            '-I./linalg/ -I./decomp/ -I./dictLearn/ dictLearn/mex/mexTrainDL_Memory.cpp',
            % compile dag toolbox
            '-I./dags/ -I./linalg/ dags/mex/mexRemoveCyclesGraph.cpp',
            '-I./dags/ -I./linalg/ dags/mex/mexCountPathsDAG.cpp',
            '-I./dags/ -I./linalg/ dags/mex/mexCountConnexComponents.cpp',
            % compile proximal toolbox
            '-I./linalg/ -I./prox/ prox/mex/mexEvalPathCoding.cpp',  
            '-I./linalg/ -I./prox/ prox/mex/mexFistaFlat.cpp',
            '-I./linalg/ -I./prox/ prox/mex/mexFistaTree.cpp',  
            '-I./linalg/ -I./prox/ prox/mex/mexFistaGraph.cpp',  
            '-I./linalg/ -I./prox/ prox/mex/mexFistaPathCoding.cpp',  
            '-I./linalg/ -I./prox/ prox/mex/mexProximalFlat.cpp', 
            '-I./linalg/ -I./prox/ prox/mex/mexProximalTree.cpp',  
            '-I./linalg/ -I./prox/ prox/mex/mexProximalGraph.cpp',
            '-I./linalg/ -I./prox/ prox/mex/mexProximalPathCoding.cpp',  
            % compile linalg toolbox
            '-I./linalg/ linalg/mex/mexCalcAAt.cpp',
            '-I./linalg/ linalg/mex/mexCalcXAt.cpp',  
            '-I./linalg/ linalg/mex/mexCalcXY.cpp',  
            '-I./linalg/ linalg/mex/mexCalcXYt.cpp', 
            '-I./linalg/ linalg/mex/mexCalcXtY.cpp',  
            '-I./linalg/ linalg/mex/mexConjGrad.cpp',  
            '-I./linalg/ linalg/mex/mexInvSym.cpp',  
            '-I./linalg/ linalg/mex/mexSort.cpp', 
            '-I./linalg/ linalg/mex/mexNormalize.cpp',  
            % compile decomp toolbox
            '-I./linalg/ -I./decomp/ decomp/mex/mexLasso.cpp',
            '-I./linalg/ -I./decomp/ decomp/mex/mexOMP.cpp',
            '-I./linalg/ -I./decomp/ decomp/mex/mexCD.cpp'
            '-I./linalg/ -I./decomp/ decomp/mex/mexL1L2BCD.cpp', 
            '-I./linalg/ -I./decomp/ decomp/mex/mexLassoMask.cpp',
            '-I./linalg/ -I./decomp/ decomp/mex/mexLassoWeighted.cpp',
            '-I./linalg/ -I./decomp/ decomp/mex/mexOMPMask.cpp',
            '-I./linalg/ -I./decomp/ decomp/mex/mexSOMP.cpp',
            '-I./linalg/ -I./decomp/ decomp/mex/mexSparseProject.cpp'};

if linux || mac
   fid=fopen('run_matlab.sh','w+');
   fprintf(fid,'#!/bin/sh\n');
end

if sixtyfourbits
   if debug
      DEFCOMMON='-largeArrayDims -DDEBUG';
   else
      DEFCOMMON='-largeArrayDims -DNDEBUG';
   end
else
   if debug
      DEFCOMMON='-DDEBUG';
   else
      DEFCOMMON='-DNDEBUG';
   end
end
if windows
   DEFCOMMON=[DEFCOMMON ' -DWINDOWS -DREMOVE_'];
end

DEFBLAS='';
if strcmp(blas,'mkl') 
   DEFBLAS='-DUSE_BLAS_LIB';
   if strcmp(arch,'GLNXA64')
      blas_link = sprintf('-Wl,--start-group %slibmkl_intel_lp64.a %slibmkl_sequential.a %slibmkl_core.a -Wl,--end-group',path_to_blas,path_to_blas,path_to_blas);
   elseif strcmp(arch,'GLNX86')
      blas_link = sprintf('-Wl,--start-group %slibmkl_intel.a %slibmkl_sequential.a %slibmkl_core.a -Wl,--end-group',path_to_blas,path_to_blas,path_to_blas);
   elseif strcmp(arch,'MACI64')
      blas_link = sprintf('%slibmkl_intel_lp64.a %slibmkl_sequential.a %slibmkl_core.a',path_to_blas,path_to_blas,path_to_blas);
   elseif strcmp(arch,'MACI') || strcmp(arch,'MAC')
      blas_link = sprintf('%slibmkl_intel.a %slibmkl_sequential.a %slibmkl_core.a',path_to_blas,path_to_blas,path_to_blas);
   elseif strcmp(arch,'PCWIN64')
      blas_link = sprintf(' -L"%s" -lmkl_intel_lp64 -lmkl_sequential -lmkl_core',path_to_blas);
   elseif strcmp(arch,'PCWIN')
      blas_link = sprintf(' -L"%s" -lmkl_intel -lmkl_sequential -lmkl_core',path_to_blas);
   else
      'unsupported achitecture'
      return;
   end
elseif strcmp(blas,'blas')
   DEFBLAS='-DUSE_BLAS_LIB';
   blas_link='-lblas -llapack';
elseif strcmp(blas,'builtin')
   blas_link='-lmwblas -lmwlapack';
   DEFBLAS='-DUSE_BLAS_LIB';
elseif strcmp(blas,'acml')
   DEFBLAS='-DUSE_BLAS_LIB -DNEW_MATLAB';
   blas_link=sprintf(' -L%s -L/usr/lib/ -lacml ',path_to_blas);
   %blas_link=sprintf(' -L%s -L/usr/lib/ -lacml -lg2c -L%s -lmv',path_to_blas,'/opt/amdsdk/v1.0/x86_open64-4.2.4/lib/gcc-lib/x86_64-open64-linux/4.2.4/');
else
   'please provide a correct blas library';
   return;
end
if ~verLessThan('matlab','7.9.0')
   DEFBLAS=[DEFBLAS ' -DNEW_MATLAB'];
end

links_lib=[blas_link];
link_flags=' -O ';

if strcmp(compiler,'icc')
   if windows
       DEFCOMP=['"PATH=%PATH%;' path_to_compiler '";"' path_to_vs '"  COMPILER=icl' ' -I"C:\Program Files (x86)\Intel\Composer XE\compiler\include"']; 
       %compile_flags='/Qvc10 /fast /QaxSSE2,SSE3,SSE4.1,SSE4.2,AVX /Qansi-alias  /Qopenmp /fp:fast=2 /MD /Oy- /GR /EHs /Zp8';
       compile_flags='/Qvc10  /Qopenmp /MD /QaxSSE2,SSE3,SSE4.1,SSE4.2,AVX,CORE-AVX2,CORE-AVX-I /O2';
   else
       DEFCOMP=sprintf('CXX=%s/icpc LDCXX=%s/icpc',path_to_compiler,path_to_compiler);
       compile_flags='-fPIC -axSSE3,SSE4.1,SSE4.2,AVX,CORE-AVX2,CORE-AVX-I -pipe -w -w0 -O3 -fomit-frame-pointer -no-prec-div -fno-alias -align -falign-functions -fp-model fast -funroll-loops ';
       %link_flags=[link_flags sprintf(' -cxxlib=%s',path_to_gcc_libraries)];
       %link_flags=[link_flags ' -gcc-version=430'];
   end
   links_lib=[links_lib ' -L"' path_to_compiler_libraries '" '];
   %links_lib=[links_lib ' -L"' path_to_compiler_libraries '" -L"' path_to_gcc_libraries '"  -lstdc++'];
   if mac
      fprintf(fid,'export LIB_INTEL=%s\n',path_to_compiler_libraries);
      fprintf(fid,sprintf('export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:%s:%s\n',path_to_compiler_libraries,path_to_blas));
      fprintf(fid,'export DYLD_INSERT_LIBRARIES=$LIB_INTEL/libimf.dylib:$LIB_INTEL/libintlc.dylib:$LIB_INTEL/libiomp5.dylib:$LIB_INTEL/libsvml.dylib\n');
   elseif linux
      fprintf(fid,'export LIB_INTEL=%s\n',path_to_compiler_libraries);
      fprintf(fid,sprintf('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s:%s\n',path_to_compiler_libraries,path_to_blas));
      fprintf(fid,'export LD_PRELOAD=$LIB_INTEL/libimf.so:$LIB_INTEL/libintlc.so.5:$LIB_INTEL/libiomp5.so:$LIB_INTEL/libsvml.so\n');
   end
   if use_multithread
      if windows
         compile_flags=[compile_flags ' /Qopenmp']; 
         %links_lib=[links_lib ' /nodefaultlib:vcomp libiomp5md.lib'];
         links_lib=[links_lib sprintf('-L"%s" -liomp5md -lmmd -lirc -lsvml_dispmd -ldecimal',path_to_compiler_libraries)];
      else
         compile_flags=[compile_flags ' -openmp'];
         links_lib=[links_lib ' -liomp5'];
         if mac || linux
            fprintf(fid,'export KMP_DUPLICATE_LIB_OK=true\n');
         end
      end
   end
elseif strcmp(compiler,'open64')
   DEFCOMP=sprintf('CXX=%s/opencc',path_to_compiler);
%   fprintf(fid,'export LIB_INTEL=%s\n',path_to_compiler_libraries);
   fprintf(fid,sprintf('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s:%s\n',path_to_compiler_libraries,path_to_blas));
   %fprintf(fid,'export LD_PRELOAD=$LIB_INTEL/libimf.so:$LIB_INTEL/libintlc.so.5:$LIB_INTEL/libiomp5.so:$LIB_INTEL/libsvml.so\n');
   compile_flags='-O2 -fomit-frame-pointer -march=opteron';
   links_lib=[links_lib ' -L"' path_to_compiler_libraries '" -L' path_to_blas ' -lstdc++'];
   if use_multithread
      compile_flags=[compile_flags ' -fopenmp'];
      links_lib=[links_lib ' -lopenmp'];
   end
elseif strcmp(compiler,'gcc')
   if windows
      DEFCOMP=['PATH=%PATH\%;' path_to_compiler ' COMPILER=g++-4' ' COMPFLAGS="-c -fexceptions" NAME_OBJECT=-o']; 
   else
      DEFCOMP=sprintf('CXX=%s/g++',path_to_compiler);
   end
   if debug
      compile_flags='-O2 -g'; 
   else
      compile_flags='-O3 -mtune=core2 -fomit-frame-pointer -funsafe-loop-optimizations'; 
   end
   links_lib=[links_lib ' -L"' path_to_compiler_libraries '" -L' path_to_blas];
   if mac
      fprintf(fid,'export LIB_GCC=%s\n',path_to_compiler_libraries);
      fprintf(fid,sprintf('export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:%s:%s\n',path_to_compiler_libraries,path_to_blas));
      fprintf(fid,'export DYLD_INSERT_LIBRARIES=$LIB_GCC/libgfortran.so:$LIB_GCC/libgcc_s.so:$LIB_GCC/libstdc++.so:$LIB_GCC/libgomp.so\n');
   elseif linux
      fprintf(fid,'export LIB_GCC=%s\n',path_to_compiler_libraries);
      fprintf(fid,sprintf('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s:%s\n',path_to_compiler_libraries,path_to_blas));
      fprintf(fid,'export LD_PRELOAD=$LIB_GCC/libgfortran.so:$LIB_GCC/libgcc_s.so:$LIB_GCC/libstdc++.so:$LIB_GCC/libgomp.so\n');
   end
   if use_multithread
      compile_flags=[compile_flags ' -fopenmp'];
      links_lib=[links_lib ' -lgomp'];
   end
elseif strcmp(compiler,'vs')
   DEFCOMP='COMPILER=cl';
   compile_flags='/c /02';
   if use_multithread
      compile_flags=[compile_flags ' /openmp'];
      links_lib=[links_lib ' -lvcomp'];
   end
elseif strcmp(compiler,'mex')
   DEFCOMP='';
   compile_flags=' -O';
   if mac || linux
      fprintf(fid,sprintf('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s\n',path_to_blas));
   end
   if use_multithread 
      if (linux || mac)
         compile_flags=[compile_flags ' -fopenmp']; % we assume gcc
         links_lib=[links_lib ' -lgomp'];
      end
   end
else
    'unknown compiler'
    return;
end

if ~windows
   fprintf(fid,'matlab $* -r \"addpath(''./build/''); addpath(''./test_release''); setenv(''MKL_NUM_THREADS'',''1''); setenv(''MKL_SERIAL'',''YES'');"\n'); 
   fclose(fid);
   !chmod +x run_matlab.sh
end

DEFS=[DEFBLAS ' ' DEFCOMMON ' ' DEFCOMP];

for k = 1:length(COMPILE),
    str = COMPILE{k};
    fprintf('compilation of: %s\n',str);
    if windows
       str = [str ' -outdir ' out_dir, ' ' DEFS ' ' links_lib ' OPTIMFLAGS="' compile_flags '" ']; 
    else
       str = [str ' -outdir ' out_dir, ' ' DEFS ' CXXOPTIMFLAGS="' compile_flags '" LDOPTIMFLAGS="' link_flags '" ' links_lib];
    end
    args = regexp(str, '\s+', 'split');
    args = args(find(~cellfun(@isempty, args)));
    mex(args{:});
end

copyfile src_release/*.m build/
