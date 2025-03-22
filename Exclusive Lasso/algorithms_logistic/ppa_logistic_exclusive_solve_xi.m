%%****************************************************************
%% solve H + sigma*A*M*At
%%****************************************************************
%% SSNAL:
%% Copyright (c) 2019 by
%% Meixia Lin, Defeng Sun, Kim-Chuan Toh, Yancheng Yuan
%%*************************************************************************
function [xi,resnrm,solve_ok,par,Ainput] = ppa_logistic_exclusive_solve_xi(Ainput,rhs,par)
resnrm = 0; solve_ok = 1;

AP = Ainput.AP;
[m,~] = size(AP);
rr1 = ~par.info_u.rr1;
D = par.info_u.D;
rho = par.sigma*par.lambda1;

sr = sum(rr1);
if sr == 0
    xi = (rhs./par.info_w.r)/par.sigdtau;
    par.innerop = 2;
    return;
end

AP1 = AP(:,rr1);
D1 = D(rr1);

%%
org_group = par.group_info.org_group;
org_group = org_group(par.group_info.P);
group = org_group(rr1);
[s,~] = sort(group);
[uniques,uidx] = unique(s);
group_num_reduced = length(uidx);
M_reduced = zeros(2,group_num_reduced);
const_vec = zeros(group_num_reduced,1);
for i = 1:group_num_reduced
    if i ~= group_num_reduced
        M_reduced(:,i) = [uidx(i);uidx(i+1)-1];
    else
        M_reduced(:,i) = [uidx(i);length(s)];
    end
    tmp = M_reduced(2,i)-M_reduced(1,i)+1;
    const_vec(i) = 2*rho/(1+tmp*2*rho);
end   

if m < 6000
    if sr < 6000
        options = 4;
    else
        options = 2;
    end
else
    options = 1;
    par.precond = 0;
end

if isfield(par,'options'); options = par.options; end

if group_num_reduced == 1
    const = const_vec(1);
    if (options == 2) || (options == 4)
        AP2 = AP1*D1;
        tmpM1 = AP1*AP1'-const*(AP2*AP2');
        tmptmp = par.info_w.r;
        M = par.sigdtau*diag(tmptmp) + par.sigma*tmpM1;
        L = mychol(M,length(M));
        xi = mylinsysolve(L,rhs);
    elseif (options == 1)
        AP2 = AP1*D1;
        par.AP1 = AP1;
        par.AP2 = AP2;
        par.const = const;
        [xi,~,resnrm,solve_ok] = psqmry('mat_ppa_logistic_exclusive_onegroup',0,rhs,par);
        par.cg.solve = solve_ok;
        par.cg.iter = length(resnrm);
    end
else
    if options == 2
        tmpM1 = 0;
        for i = 1:group_num_reduced
            Atmp = AP1(:,M_reduced(1,i):M_reduced(2,i));  
            const = const_vec(i);
            dnew = D1(M_reduced(1,i):M_reduced(2,i));
            AP2 = Atmp*dnew;
            tmpM1 = tmpM1 - (par.sigma*const)*(AP2*AP2');
        end
        tmptmp = par.info_w.r;
        M = par.sigdtau*diag(tmptmp) + par.sigma* (AP1*AP1') + tmpM1;
        L = mychol(M,length(M));
        xi = mylinsysolve(L,rhs);
    elseif options == 4
        sigM1 = zeros(sr,sr);
        for i = 1:group_num_reduced
            const = const_vec(i);
            dnew = D1(M_reduced(1,i):M_reduced(2,i));
            tmpdim = M_reduced(2,i)-M_reduced(1,i)+1;
            sigM1(M_reduced(1,i):M_reduced(2,i),M_reduced(1,i):M_reduced(2,i)) = par.sigma*speye(tmpdim,tmpdim) - (par.sigma*const)*(dnew*dnew');
        end
        tmptmp = par.info_w.r;
        M = par.sigdtau*diag(tmptmp) + AP1*sigM1*AP1';
        L = mychol(M,length(M));
        xi = mylinsysolve(L,rhs);
    elseif options == 1
        par.AP1 = AP1;
        par.D1 = D1;
        par.const_vec = const_vec;        
        par.rho = rho;
        par.counts = hist(s,uniques);
        stmp = 1:group_num_reduced;
        stmp = repelem(stmp',par.counts);
        par.smat = ind2vec(stmp',group_num_reduced);
        [xi,~,resnrm,solve_ok] = psqmry('mat_ppa_logistic_exclusive_multigroup',0,rhs,par);
        par.cg.solve = solve_ok;
        par.cg.iter = length(resnrm);
    end
end
par.innerop = options;
end

% if isfield(par,'options'); options = par.options; end
% if mm == 1
%     cr = rr1;
%     DD = D;
%     AP1 = A(:,cr);
%     dnew = DD(cr);
%     AP2 = AP1*dnew;
%     const = 2*rho/(1+sum(cr)*2*rho);
%     if (options == 2)
%         %% direct solver
%         tmpM1 = AP1*AP1'-const*(AP2*AP2');
%         tmptmp = par.info_w.r;
%         M = par.sigdtau*diag(tmptmp) + par.sigma*tmpM1;
%         L = mychol(M,length(M));
%         xi = mylinsysolve(L,rhs);
%     elseif (options == 1)
%         par.AP1 = AP1;
%         par.AP2 = AP2;
%         par.const = const;
%         [xi,~,resnrm,solve_ok] = psqmry('mat_ppa_logistic_exclusive_onegroup',0,rhs,par);
%         par.cg.solve = solve_ok;
%         par.cg.iter = length(resnrm);
%     end
% else
%     if options == 2
%         if ~isfield(Ainput,'AP')
%             AP = A(:,P);
%             Ainput.AP = AP;
%         else
%             AP = Ainput.AP;
%         end
%         tmpM1 = 0;
%         for i = 1:mm
%             Atmp = AP(:,M(1,i):M(2,i));   
%             cr = rr1(M(1,i):M(2,i));
%             DD = D(M(1,i):M(2,i)); 
%             AP1 = Atmp(:,cr);
%             dnew = DD(cr);
%             AP2 = AP1*dnew;
%             const = 2*rho/(1+sum(cr)*2*rho);
%             tmpM1 = tmpM1 + AP1*AP1'-const*(AP2*AP2');
%         end
%         tmptmp = par.info_w.r;
%         M = par.sigdtau*diag(tmptmp) + par.sigma*tmpM1;
%         L = mychol(M,length(M));
%         xi = mylinsysolve(L,rhs);
%     elseif options == 1
%         par.Amap = Ainput.Amap;
%         par.ATmap = Ainput.ATmap;
%         par.rho = rho;
%         [xi,~,resnrm,solve_ok] = psqmry('mat_ppa_logistic_exclusive_multigroup',0,rhs,par);
%         par.cg.solve = solve_ok;
%         par.cg.iter = length(resnrm);
%     end
% end
% par.innerop = options;
% end




% %%****************************************************************
% %% solve I + sigma*A*M*At
% %%****************************************************************
% %% SSNAL:
% %% Copyright (c) 2019 by
% %% Meixia Lin, Defeng Sun, Kim-Chuan Toh, Yancheng Yuan
% %%*************************************************************************
% function [xi,resnrm,solve_ok,par,Ainput] = ppa_logistic_exclusive_solve_xi(Ainput,rhs,par)
% options = 2; %options=1:CG  options=2:direct
% resnrm = 0; solve_ok = 1;
% 
% A = Ainput.A;
% m = length(rhs);
% rr1 = ~par.info_u.rr1;
% D = par.info_u.D;
% P = par.group_info.P;
% rho = par.sigma*par.lambda1;
% M = par.group_info.M;
% mm = size(M,2);
% 
% if (sum(rr1) > 5e3 && m > 5e3) || (sum(rr1) > 1.5e4 && m > 2000)
%     options = 1;
%     par.precond = 0;
% end
% 
% if sum(rr1) == 0
%     xi = rhs;
%     par.innerop = 2;
%     return;
% end
% 
% if isfield(par,'options'); options = par.options; end
% if mm == 1
%     cr = rr1;
%     DD = D;
%     AP1 = A(:,cr);
%     dnew = DD(cr);
%     AP2 = AP1*dnew;
%     const = 2*rho/(1+sum(cr)*2*rho);
%     if (options == 2)
%         %% direct solver
%         tmpM1 = AP1*AP1'-const*(AP2*AP2');
%         tmptmp = par.info_w.r;
%         M = par.sigdtau*diag(tmptmp) + par.sigma*tmpM1;
%         L = mychol(M,length(M));
%         xi = mylinsysolve(L,rhs);
%     elseif (options == 1)
%         par.AP1 = AP1;
%         par.AP2 = AP2;
%         par.const = const;
%         [xi,~,resnrm,solve_ok] = psqmry('mat_ppa_logistic_exclusive_onegroup',0,rhs,par);
%         par.cg.solve = solve_ok;
%         par.cg.iter = length(resnrm);
%     end
% else
%     if options == 2
%         if ~isfield(Ainput,'AP')
%             AP = A(:,P);
%             Ainput.AP = AP;
%         else
%             AP = Ainput.AP;
%         end
%         tmpM1 = 0;
%         for i = 1:mm
%             Atmp = AP(:,M(1,i):M(2,i));   
%             cr = rr1(M(1,i):M(2,i));
%             DD = D(M(1,i):M(2,i)); 
%             AP1 = Atmp(:,cr);
%             dnew = DD(cr);
%             AP2 = AP1*dnew;
%             const = 2*rho/(1+sum(cr)*2*rho);
%             tmpM1 = tmpM1 + AP1*AP1'-const*(AP2*AP2');
%         end
%         tmptmp = par.info_w.r;
%         M = par.sigdtau*diag(tmptmp) + par.sigma*tmpM1;
%         L = mychol(M,length(M));
%         xi = mylinsysolve(L,rhs);
%     elseif options == 1
%         par.Amap = Ainput.Amap;
%         par.ATmap = Ainput.ATmap;
%         par.rho = rho;
%         [xi,~,resnrm,solve_ok] = psqmry('mat_ppa_logistic_exclusive_multigroup',0,rhs,par);
%         par.cg.solve = solve_ok;
%         par.cg.iter = length(resnrm);
%     end
% end
% par.innerop = options;
% end
% 
