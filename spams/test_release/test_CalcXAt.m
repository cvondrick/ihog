X=randn(64,200000);
A=sprand(200,200000,0.05);

tic
XAt=mexCalcXAt(X,A);
t=toc;
fprintf('mex-file time: %fs\n',t);

tic
XAt2=X*A';
t=toc;
fprintf('mex-file time: %fs\n',t);

sum((XAt(:)-XAt2(:)).^2)
