U=randn(10,1000);

param.lambda=0.1; % regularization parameter
param.num_threads=-1; % all cores (-1 by default)
param.verbose=true;   % verbosity, false by default
param.pos=false;       % can be used with all the other regularizations
param.intercept=false; % can be used with all the other regularizations     

fprintf('First tree example\n');
% Example 1 of tree structure
% tree structured groups:
% g1= {0 1 2 3 4 5 6 7 8 9}
% g2= {2 3 4}
% g3= {5 6 7 8 9}
tree.own_variables=int32([0 2 5]);   % pointer to the first variable of each group
tree.N_own_variables=int32([2 3 5]); % number of "root" variables in each group
                              % (variables that are in a group, but not in its descendants).
                              % for instance root(g1)={0,1}, root(g2)={2 3 4}, root(g3)={5 6 7 8 9}
tree.eta_g=[1 1 1];           % weights for each group, they should be non-zero to use fenchel duality
tree.groups=sparse([0 0 0; ...
                    1 0 0; ...
                    1 0 0]);    % first group should always be the root of the tree
                                % non-zero entriees mean inclusion relation ship, here g2 is a children of g1,
                                % g3 is a children of g1

fprintf('\ntest prox tree-l0\n');                                
param.regul='tree-l0';                                
alpha=mexProximalTree(U,tree,param);

fprintf('\ntest prox tree-l2\n');                                
param.regul='tree-l2';                                
alpha=mexProximalTree(U,tree,param);

fprintf('\ntest prox tree-linf\n');                                
param.regul='tree-linf'; 
alpha=mexProximalTree(U,tree,param);

fprintf('Second tree example\n');
% Example 2 of tree structure
% tree structured groups:
% g1= {0 1 2 3 4 5 6 7 8 9}    root(g1) = { };
% g2= {0 1 2 3 4 5}            root(g2) = {0 1 2};
% g3= {3 4}                    root(g3) = {3 4};
% g4= {5}                      root(g4) = {5};
% g5= {6 7 8 9}                root(g5) = { };
% g6= {6 7}                    root(g6) = {6 7};
% g7= {8 9}                    root(g7) = {8};
% g8 = {9}                     root(g8) = {9};
tree.own_variables=  int32([0 0 3 5 6 6 8 9]);   % pointer to the first variable of each group
tree.N_own_variables=int32([0 3 2 1 0 2 1 1]); % number of "root" variables in each group
tree.eta_g=[1 1 1 2 2 2 2.5 2.5];       
tree.groups=sparse([0 0 0 0 0 0 0 0; ...
                    1 0 0 0 0 0 0 0; ...
                    0 1 0 0 0 0 0 0; ...
                    0 1 0 0 0 0 0 0; ...
                    1 0 0 0 0 0 0 0; ...
                    0 0 0 0 1 0 0 0; ...
                    0 0 0 0 1 0 0 0; ...
                    0 0 0 0 0 0 1 0]);  % first group should always be the root of the tree

fprintf('\ntest prox tree-l0\n');                                
param.regul='tree-l0';                                
alpha=mexProximalTree(U,tree,param);

fprintf('\ntest prox tree-l2\n');                                
param.regul='tree-l2'; 
alpha=mexProximalTree(U,tree,param);

fprintf('\ntest prox tree-linf\n');                                
param.regul='tree-linf'; 
alpha=mexProximalTree(U,tree,param);

% mexProximalTree also works with non-tree-structured regularization functions
fprintf('\nprox l1, intercept, positivity constraint\n');
param.regul='l1';
param.pos=true;       % can be used with all the other regularizations
param.intercept=true; % can be used with all the other regularizations     
alpha=mexProximalTree([U; ones(1,size(U,2))],tree,param);

% Example of multi-task tree
fprintf('\nprox multi-task tree\n');
param.pos=false;      
param.intercept=false;
param.lambda2=param.lambda;
param.regul='multi-task-tree';  % with linf
alpha=mexProximalTree(U,tree,param);
