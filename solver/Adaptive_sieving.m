%% Function of Path generation Using Adaptive Sieving
%% =================================================================
%% Copyright @ 2024
%% Authors: Yancheng Yuan, Meixia Lin, Defeng Sun, and Kim-Chuan Toh
%% ==================================================================
function [time,eta,obj,info] = Adaptive_sieving(Ainput,b,n,lossname,regularizer,para_list,op,solvername)
%%
npath = length(para_list);
m = length(b);
if isstruct(Ainput)
    if isfield(Ainput,'Amap'); Amap0 = Ainput.Amap; end
    if isfield(Ainput,'ATmap'); ATmap0 = Ainput.ATmap; end
else
    A = Ainput;
    Amap0 = @(x) A*x;
    ATmap0 = @(y) A'*y;
end
existA = exist('A','var');
%%
time = zeros(npath,1);
eta = zeros(npath,1);
obj = zeros(npath,1);
%%
info.nnz_list = zeros(npath,1);
info.avg_dim_list = zeros(npath,1);
info.max_dim_list = zeros(npath,1);
info.sieving_round_list = zeros(npath,1);
info.problem_cnt_list = zeros(npath,1);
%% compute the gradient of the loss function
if strcmp(lossname,'least squares')
    funf = @(x) 0.5*norm(Amap0(x)-b)^2;
    gradf = @(x) ATmap0(Amap0(x)-b);
elseif strcmp(lossname,'logistic')
    funf = @(x) sum(log(1+exp(-b.*Amap0(x))));
    gradf = @(x) ATmap0(-b./(1+exp(b.*Amap0(x))));
end
%% compute the proximal residual function
if strcmp(regularizer.name,'lasso')
    funp = @(x,lambda) lambda*sum(abs(x));
    R_fun = @(x,lambda) x - proxL1(x-gradf(x),lambda);
elseif strcmp(regularizer.name,'exclusive lasso')
    funp = @(x,lambda) lambda*xgroupnorm(x,op.group_info);
    R_fun = @(x,lambda) x - prox_exclusive(x-gradf(x),n,lambda,regularizer.group_info);
elseif strcmp(regularizer.name,'sparse group lasso')
    funp = @(x,lambda) lambda*regularizer.corg(2)*op.P.Lasso_fz(op.P.matrix*(x))+lambda*regularizer.corg(1)*sum(abs(x));
    R_fun = @(x,lambda) x - Prox_p(x-gradf(x),lambda*regularizer.corg,regularizer.P);
elseif strcmp(regularizer.name,'slope')
    funp = @(x,lambda) lambda*op.lambda_BH*sort(abs(x),'descend');
    R_fun = @(x,lambda) x - proxSortedL1(x-gradf(x),lambda*regularizer.lambda_BH');
end
%% path generation
eigsopt.issym = 1;
if isfield(op,'kmax')
    kmax = op.kmax;
else
    kmax = min(10*round(sqrt(n))+1,n);
end
stoptol = op.stoptol;
for ii = 1:npath
    ii
    lambda1 = para_list(ii);
    %%
    info.sieving_round_list(ii) = 0;
    info.problem_cnt_list(ii) = 0;
    avg_dim = 0;
    max_dim = 0;
    %% compute the index set idx
    if ii == 1
        x = sparse(n,1);
        all_var.xi = sparse(m,1);
        all_var.initial = 1;
        [~,idx] = maxk(abs(ATmap0(b)),kmax);
        idx = sort(idx,'ascend');
    else
        if strcmp(solvername,'SSNAL') 
            idx = find(x);
        else
            idx = find(abs(x)>1e-10);
        end
    end
    %% Check the current one is a solution or not.
    grad = gradf(x);
    R_value = R_fun(x,lambda1);
    eta(ii) = norm(R_value)/(1+norm(grad)+norm(x));
    if eta(ii) > stoptol
        info.sieving_round_list(ii) = info.sieving_round_list(ii) - 1;
        dimtmp = length(idx);
        avg_dim = avg_dim + dimtmp;
        max_dim = max(max_dim,dimtmp);
        while true
            tstart = clock;
            info.sieving_round_list(ii) = info.sieving_round_list(ii) + 1;
            n_reduced = length(idx);
            %% solve the reduced problem
            if n_reduced > 0
                if existA
                    A_reduced = Ainput(:,idx);
                    Ainput_reduced.A = A_reduced;
                    Ainput_reduced.Amap = @(x) A_reduced*x;
                    Ainput_reduced.ATmap = @(y) A_reduced'*y;
                else
                    Ainput_reduced.Amap = @(x) Amap0(accumarray(idx, x, [n, 1]));
                    Ainput_reduced.ATmap = @(y) feval(@(z) z(idx), ATmap0(y));  
                end
                op.Lip = eigs(@(x) Ainput_reduced.Amap(Ainput_reduced.ATmap(x)),length(b),1,'LA',eigsopt);
                %% compute essentials for reduced problems
                if strcmp(regularizer.name,'exclusive lasso')
                    op.group_info = generate_group(idx,regularizer.group_info);
                elseif strcmp(regularizer.name,'sparse group lasso')
                    op.G = 1:length(idx);
                    [unignum,gidx_temp] = unique(regularizer.sorted_group(idx));
                    op.ind = zeros(3,length(unignum));
                    for jj = 1:length(unignum)
                        op.ind(1,jj) = gidx_temp(jj);
                        if jj == length(unignum)
                            op.ind(2,jj) = length(idx);
                        else
                            op.ind(2,jj) = gidx_temp(jj+1)-1;
                        end
                    end
                    op.ind(3,:) = regularizer.ind(3,unignum);
                elseif strcmp(regularizer.name,'slope')
                    op.lambda_BH = regularizer.lambda_BH(1:n_reduced);
                end
                %%
                all_var.x = x(idx);
                all_var.u = - Ainput_reduced.ATmap(all_var.xi);
                if existA
                    all_var = inner_solver(Ainput_reduced.A,b,n_reduced,lossname,regularizer,lambda1,op,all_var,solvername);
                else
                    all_var = inner_solver(Ainput_reduced,b,n_reduced,lossname,regularizer,lambda1,op,all_var,solvername);
                end
                x = sparse(idx,1,all_var.x,n,1);
                info.problem_cnt_list(ii) = info.problem_cnt_list(ii) + 1;
            else
                x = sparse(n,1);
            end
            %% Check KKT and update index set
            grad = gradf(x);
            R_value = R_fun(x,lambda1);
            eta(ii) = norm(R_value)/(1+norm(grad)+norm(x));
            time(ii) = time(ii) + etime(clock,tstart);
            if eta(ii) < op.stoptol; break;  end
            tstart = clock;
            idx2 = find(abs(R_value) > eps);
            diff_idx = setdiff(idx2,idx);
            if isfield(op,'setkmax2')
                if isfield(op,'kmax2')
                    kmax2 = op.kmax2;
                else
                    kmax2 = max(0.1*n,2*length(idx));
                end
                if length(diff_idx) > kmax2
                    [~,sort_idx] = sort(abs(Rvalue(diff_idx)),'descend');
                    idx2 = diff_idx(sort_idx(1:kmax2));
                end
            end
            idxold = idx;
            idx = union(idx,idx2);
            if isempty(diff_idx); break;  end
            dimtmp = dimtmp + length(idx) - length(idxold);
            avg_dim = avg_dim + dimtmp;
            max_dim = max(max_dim, dimtmp);
            time(ii) = time(ii) + etime(clock,tstart);
            if time(ii) > 0.5*3600
                break;
            end
        end
    else
        dimtmp = length(idx);
        avg_dim = avg_dim + dimtmp;
        max_dim = max(max_dim,dimtmp);
        info.problem_cnt_list(ii) = 1;
    end
    obj(ii) = funf(x) + funp(x,lambda1);
    info.nnz_list(ii) = nnz(x);
    info.avg_dim_list(ii) = ceil(avg_dim/info.problem_cnt_list(ii));
    info.max_dim_list(ii) = max_dim;
end
end