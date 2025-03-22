%%*************************************************************************
%% ADMM:
%% ADMM for solving exclusive Lasso problems
%% (P) min {1/2 ||Ax - b||^2 + p(x)}
%% p(x) = lambda1 * sum_{g in G} ||x_g||_1^2
%% where lambda1 >0 is given data
%%
%% (D) max {-1/2 ||xi||^2  - <b,xi> -p^*(u) | A^T xi + u = 0}
%%*************************************************************************
%% Copyright (c) 2019 by
%% Meixia Lin, Defeng Sun, Kim-Chuan Toh, Yancheng Yuan
%%*************************************************************************
function [obj,xi,u,x,info,runhist] = admm_Exclusivelasso(Ainput,b,n,lambda1,group_info,options,x0,xi0,u0)
maxiter = 20000;
sigma   = 1;
gamma   = 1.618;
stoptol = 1e-6;
printyes = 1;
printminoryes = 1;
sig_fix = 0;
rescale = 0;
use_infeasorg = 0;
phase2 = 0;
rng('default');
Asolver = 'prox';
stopop = 2;  %2: primalkkt  %3: relgap  %4: primalkkt+relgap

if isfield(options,'sigma');    sigma = options.sigma; end
if isfield(options,'gamma');    gamma = options.gamma; end
if isfield(options,'stoptol'); stoptol = options.stoptol; end
if isfield(options,'printyes'); printyes = options.printyes; end
if isfield(options,'maxiter');  maxiter = options.maxiter; end
if isfield(options,'printminoryes'); printminoryes = options.printminoryes; end
if isfield(options,'sig_fix'); sig_fix = options.sig_fix; end
if isfield(options,'rescale'); rescale = options.rescale; end
if isfield(options,'use_infeasorg'); use_infeasorg = options.use_infeasorg; end
if isfield(options,'phase2'); phase2 = options.phase2; end
if isfield(options,'Asolver'); Asolver = options.Asolver; end

%%
%% Amap and ATmap
%%
tstart = clock;
tstart_cpu = cputime;
m = length(b); % dim of b

if (m<=5000)
    Asolver = 'direct';
else
    if (n<=5000)
        Asolver = 'woodbury';
    else 
        Asolver = 'cg';
    end
end

if isstruct(Ainput)
    if isfield(Ainput,'A'); A0 = Ainput.A; end
    if isfield(Ainput,'Amap'); Amap0 = Ainput.Amap; end
    if isfield(Ainput,'ATmap'); ATmap0 = Ainput.ATmap; end
else
    A0 = Ainput;
    Amap0 =@(x) A0*x;
    ATmap0 = @(y) A0'*y;
end
if strcmp(Asolver,'direct')
    A0A0T = A0*A0';
    AATmap0 = @(x) A0A0T*x;
else
    AATmap0 = @(x) Amap0(ATmap0(x));
end

if strcmp(Asolver,'woodbury')
    A0TA0 = A0'*A0;
end

Amap  = Amap0;
ATmap = ATmap0;
AATmap = AATmap0;
if strcmp(Asolver,'prox')
    eigsopt.issym = 1;
    rA = 2;
    [VA,dA,~] = eigs(AATmap0,m,rA,'LA',eigsopt);
    dA = diag(dA); rA = sum(dA>0);
    for i= 1:rA
        fprintf('\n %d th eigen = %3.2e',i, dA(i));
    end
    proxA = min(10,rA);
    dA = dA(1:proxA);
    VA = VA(:,1:proxA);
    VAt = VA';
    MAmap = @(xi) dA(end)*xi + VA*((dA-dA(end)).*(VAt*xi));
    MAinv = @(xi,sigma)...
        xi/(1+sigma*dA(end)) + VA(:,1:proxA)*((1./(1 + sigma*dA(1:proxA)) - 1/(1 + sigma*dA(end))).*(VAt(1:proxA,:)*xi));
end

%%
%% initiallization
borg = b;
lambda1org = lambda1;
normborg = 1 + norm(borg);
normb = normborg;
if ~exist('x0','var') || ~exist('xi0','var') || ~exist('u0','var')
    x = zeros(n,1); xi = zeros(m,1); u = zeros(n,1);
else
    x = x0; xi = xi0; u = u0;
end
bscale = 1; cscale = 1;
objscale = bscale*cscale;
if printyes
    fprintf('\n *******************************************************');
    fprintf('******************************************');
    if strcmp(Asolver,'cg')
        fprintf('\n \t\t   admm (cg)  for solving Exclusive lasso with  gamma = %6.3f', gamma);
        fprintf('\n ******************************************************');
        fprintf('*******************************************\n');
        if printminoryes
            fprintf('\n problem size: n = %3.0f, nb = %3.0f',n, m);
            fprintf('  lambda = %g', lambda1);
            fprintf('\n ---------------------------------------------------');
        end
        fprintf('\n  iter|  [pinfeas  dinfeas] [pinforg  dinforg]    relgap  |      pobj          dobj       |');
        fprintf(' time |  sigma  |gamma| [  cg  ]');
    else
        if strcmp(Asolver,'direct')
            fprintf('\n \t\t   admm (direct)  for solving Exclusive lasso with  gamma = %6.3f', gamma);
        elseif strcmp(Asolver,'woodbury')
            fprintf('\n \t\t   admm (direct, woodbury)  for solving Exclusive lasso with  gamma = %6.3f', gamma);
        elseif strcmp(Asolver,'prox')
            fprintf('\n \t\t   admm (prox)  for solving Exclusive lasso with  gamma = %6.3f', gamma);
        end
        fprintf('\n ******************************************************');
        fprintf('*******************************************\n');
        if printminoryes
            fprintf('\n problem size: n = %3.0f, nb = %3.0f',n, m);
            fprintf('  lambda = %g', lambda1);
            fprintf('\n ---------------------------------------------------');
        end
        fprintf('\n  iter|  [pinfeas  dinfeas] [pinforg  dinforg]    relgap  |      pobj          dobj       |');
        fprintf(' time |  sigma  |gamma|');
    end
end
%%
%%
Atxi = ATmap(xi); 
if strcmp(Asolver,'cg')
    AAtxi = Amap(Atxi);
    IpsigAAtxi = xi + sigma*AAtxi;
elseif strcmp(Asolver,'direct')
    AAt = A0A0T;
    Lxi = mychol(eye(m) + sigma*AAt,m);
elseif strcmp(Asolver,'woodbury')
    AtA = A0TA0;
    woodLxi = mychol((1/sigma)*eye(n) + AtA,n);
end
Ax = Amap(x);
Rp1 = Ax - b;
Rd = Atxi + u;
ARd = Amap(Rd);
primfeas = norm(Rp1 - xi)/normborg;
dualfeas = norm(Rd)/(1 + norm(u));
maxfeas = max(primfeas, dualfeas);
primfeasorg = primfeas;
dualfeasorg = dualfeas;
if printyes
    fprintf('\n initial primfeasorg = %3.2e, dualfeasorg = %3.2e', primfeasorg, dualfeasorg);
end
%%
%% main Loop
%%
breakyes = 0;
prim_win = 0;
dual_win = 0;
msg = [];
%%
for iter = 1:maxiter
    if (rescale >= 3 && rem(iter, 203)) == 0 || iter == 1
        normAtxi = norm(Atxi);
        normx = norm(x); normu = norm(u);
        normuxi = max([normAtxi,normu]);
    end
    if (((rescale == 1) && (maxfeas < 5e2) && (iter > 21) && (abs(relgap) < 0.5)) ...
            || ((rescale==2) && (maxfeas <1e-2) && (abs(relgap) < 0.05) && (iter > 40)) ...
            || ((rescale>=3) && (max(normx/normuxi,normuxi/normx) > 1.2) && (rem(iter,203)==0)))
        if (rescale <= 2)
            normAtxi = norm(Atxi);
            normx = norm(x); normu = norm(u);
            normuxi = max([normAtxi,normu]);
        end
        const = 1;
        bscale2 = normx*const;
        cscale2 = normuxi*const;
        sbc = sqrt(bscale2*cscale2);
        b = b/sbc;
        u = u/cscale2; lambda1 = lambda1/cscale2*bscale2;
        x = x/bscale2;
        xi = xi/sbc; Rp1 = Rp1/sbc;
        Amap = @(x) Amap(x*sqrt(bscale2/cscale2));
        ATmap = @(x) ATmap(x*sqrt(bscale2/cscale2));
        AATmap = @(x) AATmap(x*(bscale2/cscale2));
        Ax = Ax/sbc;
        ARd = ARd*sqrt(bscale2/cscale2^3);
        if exist('AAt','var'); AAt = (bscale2/cscale2)*AAt; end
        if exist('AtA','var'); AtA = (bscale2/cscale2)*AtA; end
        if strcmp(Asolver,'cg'); IpsigAAtxi = IpsigAAtxi/sbc; end
        if exist('Lxi','var'); Lxi = mychol(eye(m) + sigma*AAt,m); end
        if exist('woodLxi','var'); woodLxi = mychol((1/sigma)*eye(n) + AtA,n); end
        if strcmp(Asolver,'prox')
            dA = dA*bscale2/cscale2;
            MAmap = @(xi) dA(end)*xi + VA*((dA-dA(end)).*(VAt*xi));
            MAinv = @(xi,sigma)...
                xi/(1+sigma*dA(end)) + VA(:,1:proxA)*((1./(1 + sigma*dA(1:proxA)) - 1/(1 + sigma*dA(end))).*(VAt(1:proxA,:)*xi));
        end
        sigma = sigma*(cscale2/bscale2);
        
        cscale = cscale*cscale2;
        bscale = bscale*bscale2;
        objscale = objscale*(cscale2*bscale2);
        normb = 1+norm(b);
        if printyes
            fprintf('\n    ');
            fprintf('[rescale=%1.0f: %2.0f| [normx,Atxi,u =%3.2e %3.2e %3.2e] | scale=[%3.2e %3.2e]| sigma=%3.2e]',...
                rescale,iter,normx,normAtxi,normu,bscale,cscale,sigma);
        end
        rescale = rescale+1;
        prim_win = 0; dual_win = 0;
    end
    xold = x; Axold = Ax;
    %% compute xi
    if strcmp(Asolver,'cg')
        rhsxi = Rp1 - sigma*(ARd - (IpsigAAtxi - xi)/sigma);
        parxi.tol = max(0.9*stoptol,min(1/iter^1.1,0.9*maxfeas));
        parxi.sigma = sigma;
        [xi,IpsigAAtxi,resnrmxi,solve_okxi] = psqmry('matvecxi',AATmap,rhsxi,parxi,xi,IpsigAAtxi);
    elseif strcmp(Asolver,'prox')
        rhsxi = Rp1 - sigma*(ARd - MAmap(xi)); %Rp1 - sigma*(Amap(tmpxi) - MAmap(xi));
        xi = MAinv(rhsxi,sigma);
    elseif strcmp(Asolver,'direct')
        rhsxi = Rp1 - sigma*Amap(u);
        xi = mylinsysolve(Lxi,rhsxi);
    elseif strcmp(Asolver,'woodbury')
        rhsxi = Rp1 - sigma*Amap(u);
        Atrhsxi = ATmap(rhsxi);
        tmpxi = mylinsysolve(woodLxi,Atrhsxi);
        xi = rhsxi - Amap(tmpxi);
    end
    Atxi = ATmap(xi);
    %% compute u
    uinput = x - sigma*Atxi;
    up = prox_exclusive(uinput, n, sigma*lambda1, group_info);
    u = (uinput - up)/sigma;
    %% update mutilplier x
    Rd = Atxi + u;
    x = xold - gamma*sigma*Rd;
    ARd = Amap(Rd);
    Ax = Axold - gamma*sigma*ARd;
    Rp1 = Ax - b;
    %%----------------------------------------------------
    normRp = norm(Rp1 - xi);
    normRd = norm(Rd);
    normu = norm(u);
    primfeas = normRp/normb;
    dualfeas = normRd/(1+normu);
    maxfeas = max(primfeas,dualfeas);
    dualfeasorg = normRd*cscale/(1+normu*cscale);
    primfeasorg = sqrt(bscale*cscale)*normRp/normborg;
    maxfeasorg = max(primfeasorg, dualfeasorg);
    
    primobj = objscale*(0.5*norm(Rp1)^2 + lambda1*xgroupnorm(x,group_info));
    dualobj = objscale*(-0.5*norm(xi)^2 - b'*xi - pstar(u,lambda1,group_info));
    relgap = (primobj-dualobj)/( 1+abs(primobj)+abs(dualobj));
    ttime = etime(clock,tstart);
    %%-------------------------------------------------------
    %% record history
    runhist.dualfeas(iter) = dualfeas;
    runhist.primfeas(iter) = primfeas;
    runhist.maxfeas(iter) = maxfeas;
    runhist.dualfeasorg(iter) = dualfeasorg;
    runhist.primfeasorg(iter) = primfeasorg;
    runhist.maxfeasorg(iter)  = maxfeasorg;
    runhist.sigma(iter) = sigma;
    if strcmp(Asolver,'cg'); runhist.psqmrxiiter(iter) = length(resnrmxi) - 1; end
    
    %%---------------------------------------------------------
    %% check for termination
    %%---------------------------------------------------------
    if stopop == 1
        if (max([primfeasorg,dualfeasorg]) < 500*max(1e-6, stoptol))
            grad = ATmap0(Rp1*sqrt(bscale*cscale));
            etaorg = errcom(x*bscale,grad,lambda1org,n,group_info);
            eta = etaorg / (1 + norm(grad) + norm(x*bscale));
            if  eta < stoptol
                breakyes = 1;
                msg = 'converged';
            elseif (abs(relgap) < stoptol && max([primfeasorg,dualfeasorg]) < stoptol && eta < sqrt(stoptol))
                breakyes = 2;
                msg = 'converged';
            end
        end
    elseif stopop == 2
        if max([primfeasorg,dualfeasorg]) < 500*stoptol
            grad = ATmap0(Rp1*sqrt(bscale*cscale));
            etaorg = errcom(x*bscale,grad,lambda1org,n,group_info);
            eta = etaorg / (1 + norm(grad) + norm(x*bscale));
            if eta < stoptol
                breakyes = 88;
                msg = 'eta: converged';
            end
        end
    elseif stopop == 3
        if abs(relgap) < stoptol && max([primfeasorg,dualfeasorg]) < stoptol
            breakyes = 333;
            msg = 'relgap: converged';
        end
    elseif stopop == 4
        if (max([primfeasorg,dualfeasorg]) < 1e-2)
            grad = ATmap0(Rp1*sqrt(bscale*cscale));
            etaorg = errcom(x*bscale,grad,lambda1org,n,group_info);
            eta = etaorg / (1 + norm(grad) + norm(x*bscale));
            if ( (abs(relgap) < stoptol) && (max([primfeasorg,dualfeasorg]) < stoptol) && (eta < stoptol) )
                breakyes = 555;
                msg = 'eta relgap: converged';
            end
        end
    end
%     if etime(clock, tstart) > 3600
%         breakyes = 777;
%         msg = 'time out';
%     end
    
    runhist.primobj(iter)   = primobj;
    runhist.dualobj(iter)   = dualobj;
    runhist.time(iter)      = ttime;
    runhist.relgap(iter)    = relgap;
    if exist('eta','var')
        runhist.eta(iter) = eta;
    else
        runhist.eta(iter) = NaN;
    end
    %%--------------------------------------------------------
    %% print results
    %%--------------------------------------------------------
    if (iter <= 200)
        print_iter = 20;
    elseif (iter <= 2000)
        print_iter = 100;
    else
        print_iter = 200;
    end
    if (rem(iter,print_iter)==1 || iter==maxiter) || (breakyes)
        if (printyes)
            fprintf('\n %5.0d| [%3.2e %3.2e] [%3.2e %3.2e]  %- 3.2e| %- 10.7e %- 10.7e |',...
                iter,primfeas,dualfeas,primfeasorg, dualfeasorg,relgap,primobj,dualobj);
            fprintf(' %5.1f| %3.2e|',ttime, sigma);
            fprintf('%2.3f|',gamma);
            if strcmp(Asolver,'cg')
                fprintf('[%3.0d %3.0d]', length(resnrmxi)-1, solve_okxi);
            end
            if exist('eta','var')
                fprintf('\n [eta = %3.2e, etaorg = %3.2e]',...
                    eta, etaorg);
            end
        end
        if (rem(iter,5*print_iter)==1)
            normx = norm(x); normAtxi = norm(Atxi);
            normu = norm(u);
            if (printyes)
                fprintf('\n        [normx,Atxi,u =%3.2e %3.2e %3.2e]',...
                    normx,normAtxi, normu);
            end
        end
    end
    if (breakyes > 0)
        fprintf('\n  breakyes = %3.1f, %s',breakyes,msg);
        break;
    end
    
    %%-----------------------------------------------------------
    %% update sigma
    %%-----------------------------------------------------------
    if (maxfeas < 5*stoptol) %% important for nug12
        use_infeasorg = 1;
    end
    if (use_infeasorg)
        feasratio = primfeasorg/dualfeasorg;
        runhist.feasratioorg(iter) = feasratio;
    else
        feasratio = primfeas/dualfeas;
        runhist.feasratio(iter) = feasratio;
    end
    if (feasratio < 1/5)
        prim_win = prim_win+1;
    elseif (feasratio > 5)
        dual_win = dual_win+1;
    end
    sigma_update_iter = sigma_fun(iter);
    sigmascale = 1.1;%1.25;
    sigmaold = sigma;
    if (~sig_fix) && (rem(iter,sigma_update_iter)==0)
        sigmamax = 1e6; sigmamin = 1e-8;
        if (iter <= 1*2500) %% old: 1*2250
            if (prim_win > max(1,2*dual_win)) %max(1,1.2*dual_win))
                prim_win = 0;
                sigma = min(sigmamax,sigma*sigmascale);
            elseif (dual_win > max(1,2*prim_win)) %max(1,1.2*prim_win))
                dual_win = 0;
                sigma = max(sigmamin,sigma/sigmascale);
            end
        else
            if (use_infeasorg)
                feasratiosub = runhist.feasratioorg(max(1,iter-19):iter);
            else
                feasratiosub = runhist.feasratio(max(1,iter-19):iter);
            end
            meanfeasratiosub = mean(feasratiosub);
            if meanfeasratiosub < 0.1 || meanfeasratiosub > 1/0.1
                sigmascale = 1.4;
            elseif meanfeasratiosub < 0.2 || meanfeasratiosub > 1/0.2
                sigmascale = 1.35;
            elseif meanfeasratiosub < 0.3 || meanfeasratiosub > 1/0.3
                sigmascale = 1.32;
            elseif meanfeasratiosub < 0.4 || meanfeasratiosub > 1/0.4
                sigmascale = 1.28;
            elseif meanfeasratiosub < 0.5 || meanfeasratiosub > 1/0.5
                sigmascale = 1.26;
            end
            primidx = find(feasratiosub <= 1);
            dualidx = find(feasratiosub >  1);
            if (length(primidx) >= 12)
                sigma = min(sigmamax,sigma*sigmascale);
            end
            if (length(dualidx) >= 12)
                sigma = max(sigmamin,sigma/sigmascale);
            end
        end
    end
    if abs(sigmaold - sigma) > eps
        if strcmp(Asolver,'cg')
            parxi.sigma = sigma;
            AAtxi = (IpsigAAtxi - xi)/sigmaold;
            IpsigAAtxi = xi + sigma*AAtxi;
        elseif strcmp(Asolver,'direct')
            Lxi = mychol(eye(m) + sigma*AAt,m);
        elseif strcmp(Asolver,'woodbury')
            woodLxi = mychol((1/sigma)*eye(n) + AtA,n);
        end
        if strcmp(Asolver,'prox')
            MAinv = @(xi,sigma)...
                xi/(1+sigma*dA(end)) + VA(:,1:proxA)*((1./(1 + sigma*dA(1:proxA)) - 1/(1 + sigma*dA(end))).*(VAt(1:proxA,:)*xi));
        end
    end
end
%%-----------------------------------------------------------------
%% recover orignal variables
%%-----------------------------------------------------------------
if (iter == maxiter)
    msg = ' maximum iteration reached';
end
xi = xi*sqrt(bscale*cscale);
Atxi = ATmap0(xi);
x = x*bscale;
u = u*cscale;
Ax = Ax*sqrt(bscale*cscale);
Rd = Atxi + u;
Rp1 = Ax - borg;
normRp = norm(Rp1 - xi);
normRd = norm(Rd);
normu = norm(u);
primfeasorg = normRp/normborg;
dualfeasorg = normRd/(1 + normu);

primobj = 0.5*norm(Rp1)^2 + lambda1org*xgroupnorm(x,group_info);
dualobj = -(0.5*norm(xi)^2 + borg'*xi + pstar(u,lambda1,group_info));
primobjorg = 0.5*norm(Ax - borg)^2 + lambda1org*xgroupnorm(x,group_info);
relgap = (primobj-dualobj)/( 1+abs(primobj)+abs(dualobj));
obj = [primobj, dualobj];
if iter > 0
    grad = ATmap0(Ax - borg);
    etaorg = errcom(x,grad,lambda1org,n,group_info);
    eta = etaorg/(1+norm(grad)+norm(x));
else
    etaorg = nan;
    eta = nan;
end
ttime = etime(clock,tstart);
ttime_cpu = cputime - tstart_cpu;
if strcmp(Asolver,'cg'); ttCG = sum(runhist.psqmrxiiter); end
[hh,mm,ss] = changetime(ttime);
info.m = m;
info.n = n;
info.minx = min(min(x));
info.maxx = max(max(x));
info.relgap = relgap;
if strcmp(Asolver,'cg'); info.ttCG = ttCG; end
info.bscale = bscale;
info.cscale = cscale;
info.objscale = objscale;
info.iter = iter;
info.time = ttime;
info.time_cpu = ttime_cpu;
info.etaorg = etaorg;
info.eta = eta;
info.dualfeasorg = dualfeasorg;
info.primfeasorg = primfeasorg;
info.obj = obj;
info.dualfeasorg = dualfeasorg;
info.primfeasorg = primfeasorg;
info.maxfeasorg = max([dualfeasorg, primfeasorg]);
eigsopt.issym = 1;
info.Lip = eigs(AATmap0,m,1,'LA',eigsopt);


if phase2 == 1
    info.Ax = Ax;
    info.Atxi = Atxi;
end
if (printminoryes)
    if ~isempty(msg); fprintf('\n %s',msg); end
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------');
    fprintf('\n  number iter = %2.0d',iter);
    fprintf('\n  time = %3.2f,  (%d:%d:%d)',ttime,hh,mm,ss);
    if iter >= 1; fprintf('\n  time per iter = %5.4f',ttime/iter); end
    fprintf('\n  cputime = %3.2f', ttime_cpu);
    fprintf('\n     primobj = %10.9e, dualobj = %10.9e, relgap = %3.2e',primobj,dualobj, relgap);
    fprintf('\n  primobjorg = %10.9e',primobjorg);
    fprintf('\n  primfeasorg  = %3.2e, dualfeasorg = %3.2e',...
        primfeasorg, dualfeasorg);
    if (iter >= 1) && (strcmp(Asolver,'cg'))
        fprintf('\n  Total CG number = %3.0d, CG per iter = %3.1f', ttCG, ttCG/iter);
    end
    fprintf('\n  eta = %3.2e, etaorg = %3.2e', eta, etaorg);
    fprintf('\n  min(X)    = %3.2e, max(X)    = %3.2e',...
        info.minx,info.maxx);
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------\n');
end
end
%%**********************************************************************
function sigma_update_iter = sigma_fun(iter)
if (iter < 30)
    sigma_update_iter = 3;
elseif (iter < 60)
    sigma_update_iter = 6;
elseif (iter < 120)
    sigma_update_iter = 12;
elseif (iter < 250)
    sigma_update_iter = 25;
elseif (iter < 500)
    sigma_update_iter = 50;
% elseif (iter < inf)  %% better than (iter < 1000)
%     sigma_update_iter = 100;
elseif (iter < 10000)  %% better than (iter < 1000)
    sigma_update_iter = 100;
elseif (iter < inf)  %% better than (iter < 1000)
    sigma_update_iter = inf;
end
end
%%**********************************************************************

function etaorg = errcom(x,grad,lambda1,n,group_info)
tmp = x - grad;
tmp2 = prox_exclusive(tmp, n, lambda1, group_info);
etaorg = norm(x - tmp2);
end

% To change the format of time
function [h,m,s] = changetime(t)
t = round(t);
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
end
