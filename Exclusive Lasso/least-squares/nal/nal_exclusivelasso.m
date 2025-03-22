%%*************************************************************************
%% SSNAL:
%% A (dual) semismooth Newton augmented Lagrangian method for solving exclusive Lasso problems
%% (P) min {1/2 ||Ax - b||^2 + p(x)}
%% p(x) = lambda1 * sum_{g in G} ||x_g||_1^2
%% where lambda1 >0 is given data
%%
%% (D) max {-1/2 ||xi||^2  - <b,xi> -p^*(u) | A^T xi + u = 0}
%%*************************************************************************
%% SSNAL:
%% Copyright (c) 2019 by
%% Meixia Lin, Defeng Sun, Kim-Chuan Toh, Yancheng Yuan 
%%*************************************************************************
function [obj,x,xi,u,info,runhist] = nal_exclusivelasso(Ainput,b,n,lambda1,group_info,options,x0,xi0,u0)
rng('default');

iftest = 0; % test the Newton direction
maxiter = 500; % maximum iteration of the SSNAL
stoptol = 1e-6; % stop tolerance of the SSNAL
precond = 0;
stopop = 2; %2: primalkkt  %3: relgap  %4: primalkkt+relgap

printyes = 1; % print the iterations
printminoryes = 1; % print the results
printsub = 1; % print the sub iterations

startfun = @admm_Exclusivelasso;
admm.iter = 0;
admm.time = 0;
admm.timecpu = 0;

scale = 1;
rescale = 0;
runphaseI = 0;
phaseI_stoptol = 1e-3;
phaseI_maxiter = 100;


if isfield(options,'maxiter');  maxiter  = options.maxiter; end
if isfield(options,'stoptol');  stoptol  = options.stoptol; end
if isfield(options,'printyes'); printyes = options.printyes; end
if isfield(options,'printminoryes'); printminoryes = options.printminoryes; end
if isfield(options,'rescale'); rescale = options.rescale; end
if isfield(options,'Lip'); Lip = options.Lip; end
if isfield(options,'precond'); precond = options.precond; end
if isfield(options,'printsub'); printsub = options.printsub; end
if isfield(options,'runphaseI'); runphaseI = options.runphaseI; end
if isfield(options,'phaseI_stoptol'); phaseI_stoptol = options.phaseI_stoptol; end
if isfield(options,'phaseI_maxiter'); phaseI_maxiter = options.phaseI_maxiter; end
if isfield(options,'iftest'); iftest = options.iftest; end

if iftest
    printyes = 0;
    printsub = 0;
end
%% Amap and ATmap
%%
tstart = clock;
tstart_cpu = cputime;
m = length(b);
if isstruct(Ainput)
    if isfield(Ainput,'A'); A = Ainput.A; end
    if isfield(Ainput,'Amap'); Amap0 = Ainput.Amap; end
    if isfield(Ainput,'ATmap'); ATmap0 = Ainput.ATmap; end
else
    A = Ainput;
    Amap0 =@(x) A*x;
    ATmap0 = @(y) A'*y;
end
AATmap0 = @(x) Amap0(ATmap0(x));
if ~exist('Lip','var')
    eigsopt.issym = 1;
    tstartLip = clock;
    Lip = eigs(AATmap0,m,1,'LA',eigsopt);
    fprintf('\n Lip = %3.2e, time = %3.2f', Lip, etime(clock, tstartLip));
end

if exist('A','var')
    diagAAt = sum(A.*A,2);
    AP = A(:,group_info.P);
else 
    diagAAt = diagAAt_fun(Ainput,m);
end

sigmaLip = 1/sqrt(Lip);
lambda1org = lambda1;
borg = b;
normborg = 1 + norm(borg);
if ~exist('x0','var') || ~exist('xi0','var') || ~exist('u0','var')
    x = zeros(n,1); xi = zeros(m,1); u = zeros(n,1);
else
    x = x0; xi = xi0; u = u0;
end
%% phase I
admm.op.stoptol =  phaseI_stoptol;
admm.op.maxiter = phaseI_maxiter;
admm.op.sigma =  min(1,sigmaLip);
admm.op.phase2 = 1;
admm.op.use_infeasorg = 0;
admm.op.Asolver = 'prox';
admm.op.printminoryes = 0;
if admm.op.maxiter > 0 && runphaseI
    fprintf('\n Phase I: ADMM prox1 (dual approach)');
    [obj,xi,u,x,info_admm,runhist_admm] = startfun(Ainput,borg,n,lambda1org,group_info,admm.op,x,xi,u);
    admm.xi0 = xi; admm.u0 = u; admm.x0 = x;
    admm.Atxi0 = info_admm.Atxi; admm.Ax0 = info_admm.Ax;
    Atxi = admm.Atxi0; Ax = admm.Ax0;
    admm.iter = admm.iter + info_admm.iter;
    admm.time = admm.time + info_admm.time;
    admm.timecpu = admm.timecpu + info_admm.time_cpu;
    bscale = info_admm.bscale;
    cscale = info_admm.cscale;
    objscale = info_admm.objscale;
    if (info_admm.eta < stoptol)
        fprintf('\n Problem solved in Phase I \n');
        info = info_admm;
        info.m = m;
        info.n = n;
        info.minx = min(min(x));
        info.maxx = max(max(x));
        info.relgap = info_admm.relgap;
        info.iter = 0;
        info.time = admm.time;
        info.time_cpu = admm.timecpu;
        info.admmtime = admm.time;
        info.admmtime_cpu = admm.timecpu;
        info.admmiter = admm.iter;
        info.eta = info_admm.eta;
        info.etaorg = info_admm.etaorg;
        info.obj = obj;
        info.maxfeas = max([info_admm.dualfeasorg, info_admm.primfeasorg]);
        runhist = runhist_admm;
        return;
    end
else
    Atxi = ATmap0(xi); Ax = Amap0(x);
    obj(1) = 0.5*norm(Ax - borg)^2 + lambda1org*xgroupnorm(x,group_info);
    obj(2) = -(0.5*norm(xi)^2 + borg'*xi + pstar(u,lambda1org,group_info));
    bscale = 1; cscale = 1; objscale = bscale*cscale;
end
%%
if scale == 1
    b = b/sqrt(bscale*cscale);
    xi = xi/sqrt(bscale*cscale);
    Amap = @(x) Amap0(x*sqrt(bscale/cscale));
    ATmap = @(x) ATmap0(x*sqrt(bscale/cscale));
    if exist('A','var') 
        A = A*sqrt(bscale/cscale); 
        AP = AP*sqrt(bscale/cscale); 
    end
    if exist('APTAP','var')
        APTAP = APTAP*(bscale/cscale); 
    end
    diagAAt = diagAAt*(bscale/cscale);
    lambda1 = lambda1/cscale*bscale;
    x = x/bscale; u = u/cscale;
    Ax = Ax/sqrt(bscale*cscale);
    Atxi = Atxi/cscale;
    normb = 1+norm(b);
else
    Amap = @(x) Amap0(x);
    ATmap = @(x) ATmap0(x);
    normb = 1+norm(b);
end
Ainput_nal.Amap = Amap;
Ainput_nal.ATmap = ATmap;
if exist('A','var')
    Ainput_nal.A = A; 
    Ainput_nal.AP = AP;
end
if exist('APTAP','var')
    Ainput_nal.APTAP = APTAP;
end
Ainput_nal.diagAAt = diagAAt;
sigma = min(1, sigmaLip);
if isfield(options,'sigma'); sigma = options.sigma; end
Rp1 = Ax - b;
Rd  = Atxi + u;
normu = norm(u);
normRp = norm(Rp1 - xi);
normRd = norm(Rd);
primfeas = normRp/normb;
dualfeas = normRd/(1+normu);
maxfeas = max(primfeas,dualfeas);
dualfeasorg = normRd*cscale/(1+normu*cscale);
primfeasorg = sqrt(bscale*cscale)*normRp/normborg;
relgap = (obj(1) - obj(2))/(1+obj(1)+obj(2));
if printyes
    fprintf('\n *******************************************************');
    fprintf('******************************************');
    fprintf('\n \t\t   Phase II: SSNAL for solving Exclusive Lasso problem ');
    fprintf('\n ******************************************************');
    fprintf('*******************************************\n');
    fprintf('\n **************** lambda1 = %3.2e *******************',lambda1);
    if printminoryes
        fprintf('\n n = %3.0f, m = %3.0f',n, m);
        fprintf('\n bscale = %3.2e, cscale = %3.2e', bscale, cscale);
        fprintf('\n ---------------------------------------------------');
    end
    fprintf('\n  iter| [pinfeas  dinfeas]  [pinforg  dinforg]   relgaporg|      pobj           dobj      |');
    fprintf(' time |  sigma  |nnz_x');
    fprintf('\n*****************************************************');
    fprintf('**************************************************************');
    fprintf('\n %5.1d| [%3.2e %3.2e] [%3.2e %3.2e]  %- 3.2e| %- 10.7e %- 10.7e |',...
        0,primfeas,dualfeas,primfeasorg,dualfeasorg,relgap,obj(1),obj(2));
    fprintf(' %5.1f| %3.2e|',etime(clock,tstart), sigma);
end
%% ssncg
SSNCG = 1;
if SSNCG
    parNCG.sigma = sigma;
    parNCG.tolconst = 0.5;
    parNCG.n = n;
    parNCG.precond = precond;
end
maxitersub = 10;
breakyes = 0;
prim_win = 0;
dual_win = 0;
ssncgop.tol = stoptol;
ssncgop.precond = precond;
ssncgop.bscale = bscale;
ssncgop.cscale = cscale;
ssncgop.printsub = printsub;
ssncgop.iftest = iftest;
info.SSNiter = 0;
for iter = 1:maxiter
    if ((rescale == 1) && (maxfeas < 5e2) && (rem(iter,3) == 1) && (iter > 1) )...
            || ((rescale >= 2) && maxfeas < 1e-1 && (abs(relgap) < 0.05) ...
            && (iter >= 5) && (max(normx/normuxi,normuxi/normx) > 1.7) && rem(iter,5)==1)
        normAtxi = norm(Atxi);
        normx = norm(x); normu = norm(u);
        normuxi = max([normAtxi,normu]);
        if normx < 1e-7; normx = 1; normuxi = 1; end
        const = 1;
        bscale2 = normx*const;
        cscale2 = normuxi*const;
        sbc = sqrt(bscale2*cscale2);
        b = b/sbc;
        lambda1 = lambda1/cscale2*bscale2;
        x = x/bscale2;
        Ainput_nal.Amap = @(x) Ainput_nal.Amap(x*sqrt(bscale2/cscale2));
        Ainput_nal.ATmap = @(x) Ainput_nal.ATmap(x*sqrt(bscale2/cscale2));
        if isfield(Ainput_nal,'A'); Ainput_nal.A = Ainput_nal.A*sqrt(bscale2/cscale2); end
        if isfield(Ainput_nal,'AP'); Ainput_nal.AP = Ainput_nal.AP*sqrt(bscale2/cscale2); end
        if isfield(Ainput_nal,'APTAP'); Ainput_nal.APTAP = Ainput_nal.APTAP*(bscale2/cscale2); end
        Ainput_nal.diagAAt = Ainput_nal.diagAAt*(bscale2/cscale2);
        if precond == 2 && isfield(parNCG,'dA'); parNCG.dA = parNCG.dA*bscale2/cscale2; end
        xi = xi/sbc;
        Atxi = Atxi/cscale2;
        sigma = sigma*(cscale2/bscale2);
        cscale = cscale*cscale2;
        bscale = bscale*bscale2;
        objscale = objscale*(cscale2*bscale2);
        ssncgop.bscale = bscale;
        ssncgop.cscale = cscale;
        normb = 1+norm(b);
        if printyes
            fprintf('\n    ');
            fprintf('[rescale=%1.0f: %2.0f| [normx,Atxi,u=%3.2e %3.2e %3.2e] | scale=[%3.2e %3.2e]| sigma=%3.2e]',...
                rescale,iter,normx,normAtxi,normu,bscale,cscale,sigma);
        end
        rescale = rescale+1;
        prim_win = 0; dual_win = 0;
    end
    parNCG.sigma = sigma;
    parNCG.innerNT = 0;
    parNCG.innerflsa = 0;
    if dualfeas < 1e-5
        maxitersub = max(maxitersub,30);
    elseif dualfeas < 1e-3
        maxitersub = max(maxitersub,30);
    elseif dualfeas < 1e-1
        maxitersub = max(maxitersub,20);
    end
    ssncgop.maxitersub = maxitersub;
    [u,Atxi,xi,parNCG,runhist_NCG,info_NCG,Ainput_nal] = ...
        NCGExclusivelasso(b,Ainput_nal,x,Atxi,xi,lambda1,group_info,parNCG,ssncgop);
    if info_NCG.breakyes < 0
        parNCG.tolconst = max(parNCG.tolconst/1.06,1e-3);
    end
    if info_NCG.itersub > 1
        info.SSNiter = info.SSNiter+info_NCG.itersub-1;
    end
    x = info_NCG.up;
    Ax = info_NCG.Aup;
    Rd = Atxi + u;
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
    runhist.dualfeas(iter) = dualfeas;
    runhist.primfeas(iter) = primfeas;
    runhist.maxfeas(iter)  = maxfeas;
    runhist.primfeasorg(iter) = primfeasorg;
    runhist.dualfeasorg(iter) = dualfeasorg;
    runhist.maxfeasorg(iter)  = maxfeasorg;
    runhist.sigma(iter) = sigma;
    
    aver_findstep = mean(runhist_NCG.findstep);
    
    %%---------------------------------------------------------
    %% check for termination
    %%---------------------------------------------------------
    primobj = objscale*(0.5*norm(Rp1)^2 + lambda1*xgroupnorm(x,group_info));
    dualobj = objscale*(-0.5*norm(xi)^2 - b'*xi - pstar(u,lambda1,group_info));
    relgap = (primobj-dualobj)/( 1+abs(primobj)+abs(dualobj));
    ttime = etime(clock,tstart);
    if stopop == 1
        if (max([primfeasorg,dualfeasorg]) < 500*max(1e-6, stoptol))
            grad = ATmap0(Rp1*sqrt(bscale*cscale));
            etaorg = errcom(x*bscale,grad,lambda1org,n,group_info);
            eta = etaorg / (1 + norm(grad) + norm(x*bscale));
            if eta < stoptol
                breakyes = 1;
                msg = 'converged';
            elseif (abs(relgap) < stoptol && max([primfeasorg,dualfeasorg]) < stoptol && eta < sqrt(stoptol))
                breakyes = 2;
                msg = 'converged';
            end
        end
    elseif stopop == 2
        if max([primfeasorg,dualfeasorg]) < 1e-3
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
        if (max([primfeasorg,dualfeasorg]) < 1e-3)
            grad = ATmap0(Rp1*sqrt(bscale*cscale));
            etaorg = errcom(x*bscale,grad,lambda1org,n,group_info);
            eta = etaorg / (1 + norm(grad) + norm(x*bscale));
            if ( (abs(relgap) < stoptol) && (max([primfeasorg,dualfeasorg]) < stoptol) && (eta < stoptol) )
                breakyes = 555;
                msg = 'eta relgap: converged';
            end
        end
    end
    
    if etime(clock, tstart) > 3*3600
        breakyes = 777;
        msg = 'time out';
    end
    
    if (printyes)
        fprintf('\n %5.0d| [%3.2e %3.2e] [%3.2e %3.2e]  %- 3.2e| %- 10.7e %- 10.7e |',...
            iter,primfeas,dualfeas,primfeasorg, dualfeasorg,relgap,primobj,dualobj);
        fprintf(' %5.1f| %3.2e|',ttime, sigma);
        if iter >= 1
            fprintf(' %3d ',sum(abs(x) > 1e-2));
        end
        if exist('eta','var'); fprintf('\n \t [ eta = %3.2e, etaorg = %3.2e]',eta, etaorg);end
    end
    if (rem(iter,3*1)==1)
        normx = norm(x); normAtxi = norm(Atxi); normu = norm(u);
        if (printyes)
            fprintf('\n        [normx,Atxi,u =%3.2e %3.2e %3.2e ]',...
                normx,normAtxi,normu);
        end
    end
    %          fprintf('aver_findstep=%3.2e',aver_findstep);
    runhist.primobj(iter)   = primobj;
    runhist.dualobj(iter)   = dualobj;
    runhist.time(iter)      = ttime;
    runhist.relgap(iter)    = relgap;
    if exist('eta','var')
        runhist.eta(iter) = eta;
    else
        runhist.eta(iter) = NaN;
    end
    
    if (breakyes > 0)
        if printyes; fprintf('\n  breakyes = %3.1f, %s',breakyes,msg);  end
        break;
    end
    
    if (primfeasorg < dualfeasorg)
        prim_win = prim_win+1;
    else
        dual_win = dual_win+1;
    end
    if (iter < 10)
        sigma_update_iter = 1;
    elseif iter < 20
        sigma_update_iter = 2;
    elseif iter < 200
        sigma_update_iter = 2;
    elseif iter < 500
        sigma_update_iter = 3;
    end
    
    
%     if parNCG.innerop == 1
%         if aver_findstep <= 2  
%             sigmascale = 5^0.5;%2;
%         else
%             sigmascale = 2^0.5;
%         end
%     else
%         if aver_findstep <= 2 
%             sigmascale = 5;
%         else
%             sigmascale = 3;
%         end
%     end
%     sigmamax = 1e8; %5e4
     

    sigmascale = 3;   
    sigmamax = 5e4;
    
    update_sigma_options = 1;
    if update_sigma_options == 1
        if (rem(iter,sigma_update_iter)==0)
            sigmamin = 1e-4;
            if prim_win > max(1,1.2*dual_win)
                prim_win = 0;
                sigma = min(sigmamax,sigma*sigmascale);
            elseif dual_win > max(1,3*prim_win)
                dual_win = 0;
                sigma = max(sigmamin,2*sigma/sigmascale);
            end
        end
    end
end
%%-----------------------------------------------------------------
%% recover orignal variables
%%-----------------------------------------------------------------
if (iter == maxiter)
    msg = ' maximum iteration reached';
end
ttime = etime(clock,tstart);

xi = xi*sqrt(bscale*cscale);
Atxi = ATmap0(xi);
u = u*cscale;
x = x*bscale;
Ax = Ax*sqrt(bscale*cscale);
Rd = Atxi + u;
Rp1 = Ax - borg;
normRp = norm(Rp1 - xi);
normRp1 = norm(Rp1);
normRd = norm(Rd);
normu = norm(u);
primfeasorg = normRp/normborg;
dualfeasorg = normRd/(1+normu);

primobj = 0.5*norm(Rp1)^2 + lambda1org*xgroupnorm(x,group_info);
dualobj = -(0.5*norm(xi)^2 + borg'*xi + pstar(u,lambda1org,group_info));
primobjorg = 0.5*norm(Ax - borg)^2 + lambda1org*xgroupnorm(x,group_info);
relgap = (primobj-dualobj)/( 1+abs(primobj)+abs(dualobj));
obj = [primobj, dualobj];
grad = ATmap0(Rp1);
etaorg = errcom(x,grad,lambda1org,n,group_info);
eta = etaorg/(1+norm(grad)+norm(x));

ttime_cpu = cputime - tstart_cpu;
info.m = m;
info.n = n;
info.Lip = Lip;
info.minx = min(min(x));
info.maxx = max(max(x));
info.relgap = relgap;
info.iter = iter;
info.time = ttime;
[hh,mm,ss] = changetime(ttime);
info.time_cpu = ttime_cpu;
info.admmtime = admm.time;
info.admmtime_cpu = admm.timecpu;
info.admmiter = admm.iter;
info.eta = eta;
info.etaorg = etaorg;
info.obj = obj;
info.dualfeasorg = dualfeasorg;
info.primfeasorg = primfeasorg;
info.maxfeasorg = max([dualfeasorg, primfeasorg]);
info.Axmb = normRp1;
info.nnzx = calculate_nnz(x,group_info);
info.sigma = sigma;

if (printminoryes)
    if ~isempty(msg); fprintf('\n %s',msg); end
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------');
    fprintf('\n  admm iter = %3.1d, admm time = %3.1f', admm.iter, admm.time);
    fprintf('\n  number iter = %2.0d',iter);
    fprintf('\n  time = %3.2f,  (%d:%d:%d)',ttime,hh,mm,ss);
    fprintf('\n  time per iter = %5.4f',ttime/iter);
    fprintf('\n  cputime = %3.2f', ttime_cpu);
    fprintf('\n     primobj = %10.9e, dualobj = %10.9e, relgap = %3.2e',primobj,dualobj, relgap);
    fprintf('\n  primobjorg = %10.9e',primobjorg);
    fprintf('\n  primfeasorg    = %3.2e, dualfeasorg    = %3.2e',...
        primfeasorg, dualfeasorg);
    fprintf('\n  eta = %3.2e, etaorg = %3.2e', eta, etaorg);
    fprintf('\n  min(X)    = %3.2e, max(X)    = %3.2e',...
        info.minx,info.maxx);
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------\n');
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