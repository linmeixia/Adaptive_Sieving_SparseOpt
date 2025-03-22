function y = xgroupnorm(x,group_info)
y = 0;
x = x(group_info.P);
M = group_info.M;
[~,m] = size(M);
for i = 1:m
    xtmp = abs(x(M(1,i):M(2,i)));
    y = y+sum(xtmp)^2;
end
end