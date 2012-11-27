X=rand(1,300000);

tic
Y=mexSort(X);
t=toc;
toc

tic
Y2=sort(X,'ascend');
t=toc;
toc

sum((Y2(:)-Y(:)).^2)
