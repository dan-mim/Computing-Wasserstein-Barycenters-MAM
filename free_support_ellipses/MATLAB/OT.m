function [x,val,cpu] = OT(d,p,q)
tic;
m       = length(p);
n       = length(q);
beta    = [p;q];
A       = [kron(ones(1,n),speye(m)); kron(speye(n),ones(1,m))];
[x,val] = linprog_gurobi(d(:),[],[],A, beta,zeros(n*m,1));
cpu     = toc;
return