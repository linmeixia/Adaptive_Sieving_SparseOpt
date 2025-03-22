function nnz_x = calculate_nnz(x,group_info)
x = x(group_info.P);
M = group_info.M;
[~,m] = size(M);
tmp_group = zeros(m,1);
for i = 1:m
    tmp_group(i) = sum(abs(x(M(1,i):M(2,i)))>1e-5);
end
nnz_x = sum(tmp_group);
end