A=rand(1000,1000);
A=A'*A;

tic
B=mexInvSym(A);
t=toc;
fprintf('mex-file time: %fs\n',t);

tic
B2=inv(A);
t=toc;
fprintf('matlab-file time: %fs\n',t);

sum((B(:)-B2(:)).^2)
