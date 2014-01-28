% Run this script to compile the entire system.

fprintf('compiling features.cc\n');
if ispc,
    mex -O internal/features.cc -output internal/features
else
    mex -O internal/features.cc -o internal/features
end

fprintf('compiling spams\n');
run 'spams/compile.m'
