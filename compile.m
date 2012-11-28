% Run this script to compile the entire system.

mex -O features.cc -o features
run 'spams/compile.m'
