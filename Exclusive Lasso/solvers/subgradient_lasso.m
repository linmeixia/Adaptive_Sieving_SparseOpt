function [lb,ub] = subgradient_lasso(x,lambda1)
lb = sign(x);
ub = sign(x);
idx = find(x==0);
lb(idx) = -1;
ub(idx) = 1;
lb = lb*lambda1;
ub = ub*lambda1;
end