clear all; 

I=double(imread('data/lena.png'))/255;
% extract 8 x 8 patches
X=im2col(I,[8 8],'sliding');
X=X-repmat(mean(X),[size(X,1) 1]);
X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
X=X(:,1:10:end);

param.K=200;  % learns a dictionary with 100 elements
param.lambda=0.15;
param.numThreads=4; % number of threads

param.iter=100;  % let us see what happens after 100 iterations.

%%%%%%%%%% FIRST EXPERIMENT %%%%%%%%%%%
tic
D = mexTrainDL_Memory(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);

fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
ImD=displayPatches(D);
subplot(1,3,1);
imagesc(ImD); colormap('gray');
fprintf('objective function: %f\n',R);

%%%%%%%%%% SECOND EXPERIMENT %%%%%%%%%%%
tic
D = mexTrainDL(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);

fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
ImD=displayPatches(D);
subplot(1,3,2);
imagesc(ImD); colormap('gray');
fprintf('objective function: %f\n',R);
