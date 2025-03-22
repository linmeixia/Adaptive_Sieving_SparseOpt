function Sigma = generate_sparse_covariance(p,group_num, w1, w2)
num_per_g = p/group_num;
N = ceil(-16/log10(w2)+1);
% %% way 1
% tic;
% rr = [w2.^[0:N-1],zeros(1,p-N)];%w2.^[0:p-1];
% Sigma0 = toeplitz(rr);
% rrs = [w1.^[0:N-1],zeros(1,num_per_g-N)];%w1.^[0:num_per_g-1];
% Sigma_s = toeplitz(rrs);
% for i = 1:1:group_num
%     idx = [(i-1)*num_per_g+1:i*num_per_g];
%     Sigma0(idx, idx) = Sigma_s;
% end
% toc

% tic;
% N = ceil(-12/log10(w2)+1);
% rr = [w2.^[0:N-1],sparse(1,p-N)];
% Sigma1 = toeplitz(rr);
% toc
% 
% norm(Sigma1-Sigma0,'fro')


%% way2
num_idx = (2*p-N+1)*N-p;
r_idx = zeros(num_idx,1);
c_idx = zeros(num_idx,1);
value = zeros(num_idx,1);
count = 0;
for k = 2:N
    for i = 1:(p-k+1)
        for j = i+k-1
            ii = mod(i,num_per_g);
            jj = mod(j,num_per_g);
            if (ii==0) || ((ii >= num_per_g-N+2) && (jj>0)&& (jj <= ii+N-2-num_per_g+1))
                w = w2;
            else
                w = w1;
            end
            count = count+1;
            r_idx(count) = i;
            c_idx(count) = j;
            value(count) = w^(k-1);
            count = count+1;
            r_idx(count) = j;
            c_idx(count) = i;
            value(count) = w^(k-1);
        end
    end
end
for i =1:p
    count = count+1;
    r_idx(count) = i;
    c_idx(count) = i;
    value(count) = 1;
end
Sigma = sparse(r_idx,c_idx,value,p,p);
end
