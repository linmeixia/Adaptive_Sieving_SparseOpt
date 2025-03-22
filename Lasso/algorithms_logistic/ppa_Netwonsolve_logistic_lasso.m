%%****************************************************************
%% solve \nabla^2 f^* + sigma*A*M*At
%%****************************************************************
%%*************************************************************************
%% SSNAL:
%% Copyright (c) 2019 by
%% Meixia Lin, Defeng Sun, Kim-Chuan Toh, Yancheng Yuan 
%%*************************************************************************
function [dxi,resnrm,solve_ok,par] = ppa_Netwonsolve_logistic_lasso(Ainput,rhs,par)
options = 1;
resnrm = 0; solve_ok = 1;
par.Amap = Ainput.Amap;
par.ATmap = Ainput.ATmap;

if isfield(par,'options'); options = par.options; end
if (options == 1)
    %% iterative solver
    if par.precond == 1
        diagAAt = Ainput.diagAAt-diagAMAt(Ainput,par);
        invdiagM = 1./full(par.sigdtau*par.info_w.r+par.sigma*diagAAt);
        par.invdiagM = invdiagM;
    end
    [dxi,~,resnrm,solve_ok] = psqmry('mat_ppa_lasso_logistic',0,rhs,par);%numblk1
    par.cg.solve = solve_ok;
    par.cg.iter = length(resnrm);
end
par.innerop = options;

end
% %%************************************************************


function y = diagAMAt(Ainput,par)
PT = par.group_info.PT;
M = par.group_info.M;
Amap = Ainput.Amap;
D = par.info_u.D;
n = par.n;
rho = par.sigma*par.lambda1;

[~,m] = size(M);
y = 0;
for i = 1:m
    num = M(2,i)-M(1,i)+1;
    const = 2*rho/(1+2*rho*num);
    
    tmp = zeros(n,1);
    tmp(M(1,i):M(2,i)) = D{i};
    tmp = tmp(PT);
    
    xtmp = Amap(tmp);
    y = y+const*sum(xtmp.*xtmp,2);
end
end
