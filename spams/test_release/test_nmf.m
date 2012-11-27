clear all; 

I=double(imread('data/lena.png'))/255;
% extract 8 x 8 patches
X=im2col(I,[16 16],'sliding');
X=X(:,1:10:end);
X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);

param.K=49;  % learns a dictionary with 100 elements
param.numThreads=4; % number of threads

param.iter=-5;  % let us see what happens after 100 iterations.

%%%%%%%%%% FIRST EXPERIMENT %%%%%%%%%%%
tic
[U V] = nmf(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);

fprintf('Evaluating cost function...\n');
R=mean(0.5*sum((X-U*V).^2));
ImD=displayPatches(U);
imagesc(ImD); colormap('gray');
fprintf('objective function: %f\n',R);
