X=randn(64,200);
Y=randn(200,20000)';

tic
XYt=mexCalcXYt(X,Y);
t=toc;
fprintf('mex-file time: %fs\n',t);


tic
XYt2=X*Y';
t=toc;
fprintf('matlab-file time: %fs\n',t);

sum((XYt(:)-XYt2(:)).^2)
