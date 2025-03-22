%% Function of Path generation Using Warmstart
%% =================================================================
%% Copyright @ 2024
%% Authors: Yancheng Yuan, Meixia Lin, Defeng Sun, and Kim-Chuan Toh
%% ==================================================================
function [time,eta,obj] = Warmstart(Ainput,b,n,lossname,regularizer,para_list,op,solvername)
%%
npath = length(para_list);
all_var = [];
if isstruct(Ainput)
    if isfield(Ainput,'Amap'); Amap0 = Ainput.Amap; end
    if isfield(Ainput,'ATmap'); ATmap0 = Ainput.ATmap; end
else
    A = Ainput;
    Amap0 = @(x) A*x;
    ATmap0 = @(y) A'*y;
end
%%
if strcmp(lossname,'least squares')
    funf = @(x) 0.5*norm(Amap0(x)-b)^2;
    gradF = @(x) ATmap0(Amap0(x)-b);
elseif strcmp(lossname,'logistic')
    funf = @(x) sum(log(1+exp(-b.*Amap0(x))));
    gradF = @(x) ATmap0(-b./(1+exp(b.*Amap0(x))));
end
%%
if strcmp(regularizer.name,'lasso')
    funp = @(x,lambda) lambda*sum(abs(x));
    R_fun = @(x,lambda) x - proxL1(x-gradF(x),lambda);
elseif strcmp(regularizer.name,'exclusive lasso')
    funp = @(x,lambda) lambda*xgroupnorm(x,op.group_info);
    R_fun = @(x,lambda) x - prox_exclusive(x-gradF(x),n,lambda,op.group_info);
elseif strcmp(regularizer.name,'sparse group lasso')
    funp = @(x,lambda) lambda*regularizer.corg(2)*op.P.Lasso_fz(op.P.matrix*(x))+lambda*regularizer.corg(1)*sum(abs(x));
    R_fun = @(x,lambda) x - Prox_p(x-gradF(x),lambda*regularizer.corg,op.P);
elseif strcmp(regularizer.name,'slope')
    funp = @(x,lambda) lambda*op.lambda_BH*sort(abs(x),'descend');
    R_fun = @(x,lambda) x - proxSortedL1(x-gradF(x),lambda*op.lambda_BH');
end
%%
time = zeros(npath,1);
eta = zeros(npath,1);
obj = zeros(npath,1);
%%
for ii = 1:npath
    tstart = clock;
    all_var = inner_solver(Ainput,b,n,lossname,regularizer,para_list(ii),op,all_var,solvername);
    time(ii) = etime(clock,tstart);
    obj(ii) = funf(all_var.x) + funp(all_var.x,para_list(ii));
    grad = gradF(all_var.x);
    R_value = R_fun(all_var.x,para_list(ii));
    eta(ii) = norm(R_value)/(1+norm(grad)+norm(all_var.x));
end
end