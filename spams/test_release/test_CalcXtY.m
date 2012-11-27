X=randn(64,200)';
Y=randn(200,20000);

tic
XtY=mexCalcXtY(X,Y);
t=toc;
fprintf('mex-file time: %fs\n',t);

tic
XtY2=X'*Y;
t=toc;
fprintf('matlab-file time: %fs\n',t);

sum((XtY(:)-XtY2(:)).^2)
