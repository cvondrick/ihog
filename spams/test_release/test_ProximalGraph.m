U=randn(10,1000);

param.lambda=0.1; % regularization parameter
param.num_threads=-1; % all cores (-1 by default)
param.verbose=true;   % verbosity, false by default
param.pos=false;       % can be used with all the other regularizations
param.intercept=false; % can be used with all the other regularizations     

fprintf('First graph example\n');
% Example 1 of graph structure
% groups:
% g1= {0 1 2 3}
% g2= {3 4 5 6}
% g3= {6 7 8 9}
graph.eta_g=[1 1 1];
graph.groups=sparse(zeros(3));
graph.groups_var=sparse([1 0 0; 
                         1 0 0; 
                         1 0 0; 
                         1 1 0; 
                         0 1 0;
                         0 1 0;
                         0 1 1;
                         0 0 1;
                         0 0 1;
                         0 0 1]);

fprintf('\ntest prox graph\n');                                
param.regul='graph';
alpha=mexProximalGraph(U,graph,param);

% Example 2 of graph structure
% groups:
% g1= {0 1 2 3}
% g2= {3 4 5 6}
% g3= {6 7 8 9}
% g4= {0 1 2 3 4 5}
% g5= {6 7 8}
graph.eta_g=[1 1 1 1 1];
graph.groups=sparse([0 0 0 1 0;
                     0 0 0 0 0;
                     0 0 0 0 0;
                     0 0 0 0 0;
                     0 0 1 0 0]);   % g5 is included in g3, and g2 is included in g4
graph.groups_var=sparse([1 0 0 0 0; 
                         1 0 0 0 0; 
                         1 0 0 0 0 ; 
                         1 1 0 0 0; 
                         0 1 0 1 0;
                         0 1 0 1 0;
                         0 1 0 0 1;
                         0 0 0 0 1;
                         0 0 0 0 1;
                         0 0 1 0 0]); % represents direct inclusion relations 
                                      % between groups (columns) and variables (rows)

fprintf('\ntest prox graph\n');                                
param.regul='graph';
alpha=mexProximalGraph(U,graph,param);

fprintf('\ntest prox multi-task-graph\n');                                
param.regul='multi-task-graph';
param.lambda2=0.1;
alpha=mexProximalGraph(U,graph,param);

fprintf('\ntest no regularization\n');                                
param.regul='none';
alpha=mexProximalGraph(U,graph,param);
