%% Test file for Exclusive lasso logistic regression model
%% =================================================================
%% Copyright @ 2025
%% Authors: Yancheng Yuan, Meixia Lin, Defeng Sun, and Kim-Chuan Toh
%% ==================================================================
clear all
%% function form
lossname = 'logistic';
regularizer.name = 'exclusive lasso';
%%
run_as = 1;
run_warmstart = 1;
%%
n_A = [500];
group_num = 20;
p_list = [5000];
%%
Table = cell(2+length(n_A)*length(p_list),7);
Table{1,3} = 'Total time (hh:mm:ss)';
Table{1,5} = 'Information of the AS';
Table{2,1} = 'm';
Table{2,2} = 'n(=gxp)';
Table{2,3} = 'Warmstart';
Table{2,4} = 'With AS';
Table{2,5} = 'Sieving_round';
Table{2,6} = 'Avg_dim';
Table{2,7} = 'Max_dim';
%%
changetime = @(t) ([num2str(floor(round(t)/3600)),':',num2str(floor(rem(round(t),3600)/60)),':',num2str(rem(rem(round(t),60),60))]);
nnz_ratio = 0.001;
for kk = 1:length(n_A)
    Table{3+(kk-1)*length(p_list),1} = n_A(kk);
    for i = 1:length(p_list)
        %% Load data
        clearvars -except Table n_A group_num p_list kk i nnz_ratio run_as run_warmstart changetime lossname regularizer;
        if strcmp(lossname,'least squares')
            Data = generate_sparse_Allen(n_A(kk),group_num*p_list(i),group_num, 0.9, 0.3, nnz_ratio*p_list(i));
        elseif strcmp(lossname,'logistic')
            Data = generate_Allen_logistic(n_A(kk),p_list(i),group_num, 0.9, 0.3, nnz_ratio);
        end
        fprintf('\n Data construction completed for m = %d, n = %d \n', n_A(kk), group_num*p_list(i));
        A = Data.A;
        A = A-mean(A,2);
        A = A./sqrt(sum(A.*A));
        b = Data.y;
        group_info = Data.group_info;
        n = size(A,2);
        clear Data;
        %% compute essentials for regularized problems
        [regularizer,op] = compute_info_of_regularizer(A,n,group_info,regularizer);
        %%
        eigsopt.issym = 1;
        op.Lip = eigs(@(x) A*((x'*A)'),length(b),1,'LA',eigsopt);
        %%
        lambda_max = norm(A'*b,'inf');
        npath = 20;
        ratio_max = 1e-1;
        ratio_min = 1e-4;
        logratio = log(ratio_min/ratio_max);
        delta = logratio/(npath-1);
        log_lratio = 0:delta:logratio;
        lambda1_list = exp(log_lratio)*lambda_max*ratio_max;
        %%
        op.stoptol = 1e-6;
        op.runphaseI = 0;
        %%
        if run_warmstart
            [warmstart_time] = Warmstart(A,b,n,lossname,regularizer,lambda1_list,op,'SSNAL');
            warmstart_time = sum(warmstart_time);
        else
            warmstart_time = 0;
        end
        %%
        if run_as
            [with_as_time,~,~,AS_info] = Adaptive_sieving(A,b,n,lossname,regularizer,lambda1_list,op,'SSNAL');
            with_as_time = sum(with_as_time);
        else
            with_as_time = 0;
        end
        Table{2+(kk-1)*length(p_list)+i,2} = group_num*p_list(i);
        Table{2+(kk-1)*length(p_list)+i,3} = changetime(warmstart_time);
        Table{2+(kk-1)*length(p_list)+i,4} = changetime(with_as_time);
        Table{2+(kk-1)*length(p_list)+i,5} = sum(AS_info.sieving_round_list);
        Table{2+(kk-1)*length(p_list)+i,6} = ceil(sum(AS_info.avg_dim_list.*AS_info.problem_cnt_list)/sum(AS_info.problem_cnt_list));
        Table{2+(kk-1)*length(p_list)+i,7} = max(AS_info.max_dim_list);
    end
end