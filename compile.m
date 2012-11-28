% Run this script to compile the entire system.

fprintf('compiling features.cc\n');
mex -O internal/features.cc -o internal/features

fprintf('compiling spams\n'):
run 'spams/compile.m'
