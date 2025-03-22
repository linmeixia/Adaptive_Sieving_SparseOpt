function Data = generate_Allen_logistic(n,num_per_g,group_num, w1, w2, nnz_r)
p = num_per_g*group_num;
Sigma0 = generate_sparse_covariance(p,group_num, w1, w2);
[R,indef] = chol(Sigma0);
if (indef); error('Sigma0 is not positive definite'); end
Data.A = randn(n,p)*R;
num_nonzeros = ceil(num_per_g*nnz_r);
x = sparse(p,1);
for i =1:1:group_num
    x_temp = zeros(num_per_g,1);
    idx_temp = randperm(num_per_g, num_nonzeros);
    x_temp(idx_temp) = rand(num_nonzeros,1)*10;
    x((i-1)*num_per_g+1:i*num_per_g) = x_temp;
end
group_M = zeros(2, group_num);
for i = 1:1:group_num
    group_M(1, i) = 1+num_per_g*(i-1);
    group_M(2, i) = num_per_g*i;
end
group_info.M = group_M;
group_info.group_num = group_num;
%% Shuttle my variables
if group_num~=1
    group_info.PT = randperm(p);
    [~, group_info.P] = sort(group_info.PT);
    x = x(group_info.PT);
    Data.A = Data.A(:, group_info.PT);
else
    group_info.P = [1:1:p];
    group_info.PT = [1:1:p];
end
y_now = Data.A*x + randn(n, 1);
Data.y = ones(n,1);
Data.y(y_now<0) = -1;

Data.groud_truth = x;
Data.p = p;
Data.n = n;
Data.group_info = group_info;
end