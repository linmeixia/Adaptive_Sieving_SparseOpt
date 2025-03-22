%% Generate a test data for exclusive lasso
%% A \in R^{n \times p}: n samples , p features
%% k groups(group_num), each group has one useful features

function Data = generate_sparse_Allen(n,p,group_num, w1, w2, num_nonzeros)
num_per_g = p/group_num;
Sigma0 = generate_sparse_covariance(p,group_num, w1, w2);
[R,indef] = chol(Sigma0);
if (indef); error('Sigma0 is not positive definite'); end
A = randn(n,p)*R;

x = zeros(p,1);
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
if group_num ~= 1
    group_info.PT = randperm(p);
    [~, group_info.P] = sort(group_info.PT);
    x = x(group_info.PT);
    A = A(:, group_info.PT);
else 
    group_info.P = [1:1:p];
    group_info.PT = [1:1:p];
end
org_group = zeros(n,1);
for i = 1:size(group_info.M,2)
    org_group(group_info.P(group_info.M(1,i):group_info.M(2,i))) = i;
end
group_info.org_group = org_group;
y = A*x + randn(n, 1);
Data.A = A;
Data.y = y;
Data.ground_truth = x;
Data.p = p;
Data.n = n;
Data.group_info = group_info;
end
