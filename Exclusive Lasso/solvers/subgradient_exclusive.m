function [lb,ub] = subgradient_exclusive(x,n,lambda1,group_info)
lb = zeros(n,1);
ub = zeros(n,1);
x = x(group_info.P);
M = group_info.M;
[~,m] = size(M);
for i = 1:m
    xtmp = x(M(1,i):M(2,i));
    tmp2 = 2*sum(abs(xtmp));
    lbtmp = sign(xtmp);
    ubtmp = sign(xtmp);
    idx = find(xtmp==0);
    lbtmp(idx) = -1;
    ubtmp(idx) = 1;
    lbtmp = lbtmp*(tmp2*lambda1);
    ubtmp = ubtmp*(tmp2*lambda1);
    lb(M(1,i):M(2,i)) = lbtmp;
    ub(M(1,i):M(2,i)) = ubtmp;
end
lb = lb(group_info.PT);
ub = ub(group_info.PT);
end

