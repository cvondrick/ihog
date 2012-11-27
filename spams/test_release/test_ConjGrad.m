A=randn(5000,500);
A=A'*A;
b=ones(500,1);
x0=b;
tol=1e-4;
itermax=0.5*length(b);

tic
for ii = 1:20
x1 = mexConjGrad(A,b,x0,tol,itermax);
end
t=toc;
fprintf('mex-file time: %fs\n',t);

tic
for ii = 1:20
x2 = pcg(A,b);
end
t=toc;
fprintf('Matlab time: %fs\n',t);
sum((x1(:)-x2(:)).^2)
