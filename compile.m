% Run this script to compile the entire system.

fprintf('compiling computeHOG.cc\n');
if ispc,
    mex -O internal/computeHOG.cc -output internal/computeHOG
else
    mex -O internal/computeHOG.cc -o internal/computeHOG
end

fprintf('compiling spams\n');
run 'spams/compile.m'
