X=randn(64,200);
Y=randn(200,20000);

tic
XY=mexCalcXY(X,Y);
t=toc;
fprintf('mex-file time: %fs\n',t);

tic
XY2=X*Y;
t=toc;
fprintf('mex-file time: %fs\n',t);

sum((XY(:)-XY2(:)).^2)
