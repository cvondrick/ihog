A=sprand(200,200000,0.05);

tic
AAt=mexCalcAAt(A);
t=toc;
fprintf('mex-file time: %fs\n',t);

tic
AAt2=A*A';
t=toc;
fprintf('matlab time: %fs\n',t);

sum((AAt(:)-AAt2(:)).^2)
