%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Use the ADMM to solve the Lasso linear regression problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [obj,y,xi,x,info,runhist] = admmL1S(Ainput,b,n,lambda,options,y0,xi0,x0)
maxiter = 20000;
gamma   = 1.618;
stoptol = 1e-6;
printyes = 1;
printminoryes = 1;
dscale = ones(n,1);
sig_fix = 0;
rescale = 1;
use_infeasorg = 0;
phase2 = 0;
rng('default');
Asolver = 'prox1';
Ascale = 1;
Ascaleyes = 0;

if isfield(options,'gamma');    gamma = options.gamma; end
if isfield(options,'stoptol'); stoptol = options.stoptol; end
if isfield(options,'printyes'); printyes = options.printyes; end
if isfield(options,'maxiter');  maxiter = options.maxiter; end
if isfield(options,'printminoryes'); printminoryes = options.printminoryes; end
if isfield(options,'sig_fix'); sig_fix = options.sig_fix; end
if isfield(options,'dscale'); dscale = options.dscale; end
if isfield(options,'rescale'); rescale = options.rescale; end
if isfield(options,'use_infeasorg'); use_infeasorg = options.use_infeasorg; end
if isfield(options,'phase2'); phase2 = options.phase2; end
if isfield(options,'Lip'); Lip = options.Lip; end
if isfield(options,'Asolver'); Asolver = options.Asolver; end
if isfield(options,'Ascale');  Ascale = options.Ascale; end
%%
%% Amap and ATmap
%%
tstart = clock;
tstart_cpu = cputime;
if isstruct(Ainput)
    if isfield(Ainput,'A'); A0 = Ainput.A; end
    if isfield(Ainput,'Amap'); Amap0 = Ainput.Amap; end
    if isfield(Ainput,'ATmap'); ATmap0 = Ainput.ATmap; end
else
    A0 = Ainput;
    Amap0 =@(x) A0*x;
    ATmap0 = @(y) A0'*y;
end

if Ascale == 1 && exist('A0','var')
    dscale = 1./max(1,sqrt(sum(A0.*A0))');
    Amap = @(x) Amap0(dscale.*x);
    ATmap = @(x) dscale.*ATmap0(x);
    A = A0*spdiags(dscale,0,n,n);
    Ascaleyes = 1;
else
    Amap  = Amap0;
    ATmap = ATmap0;
    A = A0;
end
AATmap = @(x) Amap(ATmap(x));

if ~exist('Lip','var') || Ascaleyes
    eigsopt.issym = 1;
    eigsopt.v0 = randn(length(b),1);
    Lip = eigs(AATmap,length(b),1,'LA',eigsopt);
    fprintf('\n Lip = %3.2e', Lip);
end
%%
%% initiallization
m = length(b); % dim of b
borg = b;
if norm(dscale - 1) < eps
    ld = lambda;
else
    ld = lambda*dscale;
end
lambdaorg = lambda;
normborg = 1 + norm(borg);
normb = normborg;
sigma = max(1e-4,min([1,lambdaorg,1/Lip]));
if Ascaleyes; sigma = 1; end %sigma/mean(dscale)^2; end
if isfield(options,'sigma');    sigma = options.sigma; end
if exist('A0','var') && strcmp(Asolver,'direct')
    AAt = A*A';
end
if ~exist('x0','var') || ~exist('xi0','var') || ~exist('y0','var')
    x = zeros(n,1); xi = zeros(m,1); y = x;
else
    x = x0; xi = xi0; y = y0;
    if Ascaleyes; y = y.*dscale; end
end
bscale = 1; cscale = 1;
objscale = bscale*cscale;
if printyes
    fprintf('\n *******************************************************');
    fprintf('******************************************');
    fprintf('\n \t\t   admm  for solving lasso with  gamma = %6.3f', gamma);
    fprintf('\n ******************************************************');
    fprintf('*******************************************\n');
    if printminoryes
        fprintf('\n problem size: n = %3.0f, nb = %3.0f',n, m);
        fprintf('  lambda = %g ', lambda);
        fprintf('\n ---------------------------------------------------');
    end
    fprintf('\n  iter|  [pinfeas  dinfeas] [pinforg dinforg]   relgap |    pobj       dobj      |');
    fprintf(' time |  sigma  |gamma |');
end
%%
%%
Atxi = ATmap(xi); AAtxi = Amap(Atxi);
if strcmp(Asolver,'cg')
    IpsigAAtxi = xi + sigma*AAtxi;
elseif strcmp(Asolver,'direct')
    if m > 200
        Lxi = mychol(eye(m) + sigma*AAt,m);
    end
elseif strcmp(Asolver,'prox1')
    rho = Lip;
end
Ax = Amap(x);
Rp1 = Ax - b;
Rp = Rp1 + xi;
Rd = Atxi + y;
ARd = Amap(Rd);
primfeas = norm(Rp)/normborg;
dualfeas = norm(Rd,'fro')/(1+norm(y));
maxfeas = max(primfeas, dualfeas);
primfeasorg = primfeas;
dualfeasorg = norm(Rd./dscale,'fro')/(1+norm(y./dscale));
runhist.cputime(1) = etime(clock,tstart);
runhist.psqmrxiiter(1) = 0;
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
        normy = norm(y); normAtxi = norm(Atxi);
        normx = norm(x);
        normyxi = max(normy,normAtxi);
    end
    if (((rescale == 1) && (maxfeas < 5e2) && (iter > 21) && (abs(relgap) < 0.2)) ...
            || ((rescale==2) && (maxfeas <1e-2) && (abs(relgap) < 0.05) && (iter > 40)) ...
            || ((rescale>=3) && (max(normx/normyxi,normyxi/normx) > 1.2) && (rem(iter,203)==0)))
        if (rescale <= 2)
            normy = norm(y); normAtxi = norm(Atxi);
            normx = norm(x);
            normyxi = max(normy,normAtxi);
        end
        const = 1;
        bscale2 = normx*const;
        cscale2 = normyxi*const;
        sbc = sqrt(bscale2*cscale2);
        b = b/sbc;
        y = y/cscale2; ld = ld/cscale2; lambda = lambda/cscale2;
        x = x/bscale2;
        Amap = @(x) Amap(x*sqrt(bscale2/cscale2));
        ATmap = @(x) ATmap(x*sqrt(bscale2/cscale2));
        AATmap = @(x) AATmap(x*(bscale2/cscale2));
        Ax = Ax/sbc;
        if exist('AAt','var')
            AAt = (bscale2/cscale2)*AAt;
        end
        if strcmp(Asolver,'cg')
            IpsigAAtxi = IpsigAAtxi/sbc;
        end
        if strcmp(Asolver,'prox1')
            rho = rho*bscale2/cscale2;
        end
        xi = xi/sbc; Rp1 = Rp1/sbc;
        ARd = ARd*sqrt(bscale2/cscale2^3);
        sigma = sigma*(cscale2/bscale2);
        cscale = cscale*cscale2;
        bscale = bscale*bscale2;
        objscale = objscale*(cscale2*bscale2);
        normb = 1+norm(b);
        if printyes
            fprintf('\n    ');
            fprintf('[rescale=%1.0f: %2.0f| %3.2e %3.2e %3.2e | %3.2e %3.2e| %3.2e]',...
                rescale,iter,normx,normAtxi, normy,bscale,cscale,sigma);
        end
        rescale = rescale+1;
        prim_win = 0; dual_win = 0;
    end
    xold = x; Axold = Ax;
    %% compute xi

    if strcmp(Asolver,'cg')
        rhsxi = -Rp1 - sigma*(ARd - (IpsigAAtxi - xi)/sigma);
        parxi.tol = max(0.9*stoptol,min(1/iter^1.1,0.9*maxfeas));
        parxi.sigma = sigma;
        [xi,IpsigAAtxi,resnrmxi,solve_okxi] = psqmry('matvecxi',AATmap,rhsxi,parxi,xi,IpsigAAtxi);
    elseif strcmp(Asolver,'direct')
        rhsxi = -Rp1 - sigma*Amap(y);
        if m<=200
            xi = (eye(m) + sigma*AAt)\rhsxi;
        else
            xi = mylinsysolve(Lxi,rhsxi);
        end
    elseif strcmp(Asolver,'prox1')
        rhsxi = -Rp1 - sigma*(ARd - rho*xi);
        xi = rhsxi/(1+sigma*rho);
    end
    Atxi = ATmap(xi);
    %% compute y
    yinput = -Atxi - x/sigma;
    [y,rr] = proj_inf(yinput,ld);
    %% update mutilplier Xi, y
    Rd = Atxi + y;
    x  = xold + gamma*sigma*Rd;
    ARd = Amap(Rd);
    Ax = Axold + gamma*sigma*ARd;
    %%----------------------------------------------------
    normRd = norm(Rd); normy = norm(y);
    dualfeas = normRd/(1+normy);
    if Ascaleyes
        dualfeasorg = norm(Rd./dscale)*cscale/(1+norm(y./dscale)*cscale);
    else
        dualfeasorg = normRd*cscale/(1 + normy*cscale);
    end
    Rp1 = Ax - b;
    Rp = Rp1 + xi;
    normRp = norm(Rp);
    primfeas = normRp/normb;
    primfeasorg = sqrt(bscale*cscale)*normRp/normborg;
    maxfeas = max(primfeas,dualfeas);
    maxfeasorg = max(primfeasorg, dualfeasorg);
    %% record history
    runhist.dualfeas(iter) = dualfeas;
    runhist.primfeas(iter) = primfeas;
    runhist.dualfeasorg(iter) = dualfeasorg;
    runhist.primfeasorg(iter) = primfeasorg;
    runhist.maxfeasorg(iter)  = maxfeasorg;
    runhist.sigma(iter) = sigma;
    if strcmp(Asolver,'cg'); runhist.psqmrxiiter(iter) = length(resnrmxi) - 1; end
    runhist.rr(iter) = sum(1- rr);
    runhist.xr(iter) = sum(abs(x)>1e-10);
    %%---------------------------------------------------------
    %% check for termination
    %%---------------------------------------------------------
    if (max([primfeasorg,dualfeasorg]) < 1e2*stoptol)
        grad = ATmap0(Rp1*sqrt(bscale*cscale));
        if Ascaleyes
            etaorg = norm(grad + proj_inf(dscale.*x*bscale - grad,lambdaorg));
            eta = etaorg/(1+norm(grad)+norm(dscale.*x*bscale));
        else
            etaorg = norm(grad + proj_inf(x*bscale - grad,lambdaorg));
            eta = etaorg/(1+norm(grad)+norm(x*bscale));
        end
        if eta < stoptol
            breakyes = 1;
            msg = 'converged';
        end
    end
    %%---------------------------------------------------------
    if phase2
        if iter>= 2
            if iter<=5 && (runhist.xr(iter - 1) <= runhist.xr(iter))
                if iter >= 3 && max(runhist.xr(1:iter)) <= 1
                    breakyes = 888;
                    msg = 'pahse1 to phase2: case 1';
                end
            elseif rem(iter,25) == 0
                idxtmp1 = [iter-10:iter];
                if mean(runhist.psqmrxiiter) <= 10 && ...
                        (abs(mean(runhist.xr(idxtmp1)) - mean(runhist.xr(idxtmp1-11))) <= 10 ...
                        || abs(mean(runhist.rr(idxtmp1)) - mean(runhist.rr(idxtmp1-11))) <= 10)
                    %maxiter = iter;
                    breakyes = 999;
                    msg = 'pahse1 to phase2: case 2';
                end
            end
        end
    end

    if etime(clock, tstart) > 0.5*3600  %7*3600
        breakyes = 777;
        msg = 'time out';
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
    if (rem(iter,print_iter)==1 || iter==maxiter) || (breakyes) || iter < 20
        primobj = objscale*(0.5*norm(Rp1)^2 + norm(ld.*x,1));
        dualobj = objscale*(-0.5*norm(xi)^2 + b'*xi);
        relgap = (primobj-dualobj)/( 1+abs(primobj)+abs(dualobj));
        ttime = etime(clock,tstart);
        if (printyes)
            fprintf('\n %5.0d| [%3.2e %3.2e] [%3.2e %3.2e]  %- 3.2e| %- 5.4e %- 5.4e |',...
                iter,primfeas,dualfeas,primfeasorg, dualfeasorg,relgap,primobj,dualobj);
            fprintf(' %5.1f| %3.2e|',ttime, sigma);
            fprintf('%2.3f|',gamma);
            if strcmp(Asolver,'cg')
                fprintf('[%3.0d %3.0d]', length(resnrmxi)-1, solve_okxi);
            end
            fprintf('[%3d, %3d]|',runhist.rr(iter),runhist.xr(iter));
            if exist('eta','var'); fprintf('[eta = %3.2e, etaorg = %3.2e]',eta, etaorg); end
        end
        if (rem(iter,5*print_iter)==1)
            normx = norm(x); normAtxi = norm(Atxi); normy = norm(y);
            if (printyes)
                fprintf('\n        [normx,Atxi,y =%3.2e %3.2e %3.2e]',...
                    normx,normAtxi,normy);
            end
        end
        runhist.primobj(iter)   = primobj;
        runhist.dualobj(iter)   = dualobj;
        runhist.time(iter)      = ttime;
        runhist.relgap(iter)    = relgap;
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
    if Ascaleyes; use_infeasorg = 1; end
    if (use_infeasorg)
        feasratio = primfeasorg/dualfeasorg;
        runhist.feasratioorg(iter) = feasratio;
    else
        feasratio = primfeas/dualfeas;
        runhist.feasratio(iter) = feasratio;
    end
    if (feasratio < 1)
        prim_win = prim_win+1;
    else
        dual_win = dual_win+1;
    end
    sigma_update_iter = sigma_fun(iter);
    sigmascale = 1.25;
    sigmaold = sigma;
    if (~sig_fix) && (rem(iter,sigma_update_iter)==0)
        sigmamax = 1e6; sigmamin = 1e-4;
        if (iter <= 1*2500) %% old: 1*2250
            if (prim_win > max(1,1.2*dual_win))
                prim_win = 0;
                sigma = min(sigmamax,sigma*sigmascale);
            elseif (dual_win > max(1,1.2*prim_win))
                dual_win = 0;
                sigma = max(sigmamin,sigma/sigmascale);
            end
        end
    end
    if abs(sigmaold - sigma) > eps
        if strcmp(Asolver,'cg')
            parxi.sigma = sigma;
            AAtxi = (IpsigAAtxi - xi)/sigmaold;
            IpsigAAtxi = xi + sigma*AAtxi;
        elseif exist('Lxi','var')
            Lxi = mychol(eye(m) + sigma*AAt,m);
        end
    end
end
%%-----------------------------------------------------------------
%% recover orignal variables
%%-----------------------------------------------------------------
if (iter == maxiter)
    msg = ' maximum iteration reached';
    info.termcode = 3;
end
xi = xi*sqrt(bscale*cscale);
Atxi = ATmap0(xi);
y = y*cscale;
x = x*bscale;
if Ascaleyes; x = dscale.*x; y = y./dscale; end
ttime = etime(clock,tstart);
ttime_cpu = cputime - tstart_cpu;
Rd = Atxi + y;
dualfeasorg = norm(Rd)/(1+norm(y));
Ax = Ax*sqrt(bscale*cscale);
Rp = Ax - borg + xi;
primfeasorg = norm(Rp)/normborg;
primobj = 0.5*norm(Ax - borg)^2 + lambdaorg*norm(x,1);
dualobj = -0.5*norm(xi)^2 + borg'*xi;
relgap = (primobj-dualobj)/( 1+abs(primobj)+abs(dualobj));
obj = [primobj, dualobj];
grad = ATmap0(Ax - borg);
etaorg = norm(grad + proj_inf(x - grad,lambdaorg));
eta = etaorg/(1+norm(grad)+norm(x));
runhist.m = m;
runhist.n = n;
ttCG = sum(runhist.psqmrxiiter);
runhist.iter = iter;
runhist.totaltime = ttime;
runhist.primobjorg = primobj;
runhist.dualobjorg = dualobj;
runhist.maxfeas = max([dualfeasorg, primfeasorg]);
runhist.eta = eta;
runhist.etaorg = etaorg;
info.m = m;
info.n = n;
info.minx = min(min(x));
info.maxx = max(max(x));
info.relgap = relgap;
info.ttCG = ttCG;
info.iter = iter;
info.time = ttime;
info.time_cpu = ttime_cpu;
info.sigma = sigma;
info.eta = eta;
info.etaorg = etaorg;
info.bscale = bscale;
info.cscale = cscale;
info.objscale = objscale;
info.dualfeasorg = dualfeasorg;
info.primfeasorg = primfeasorg;
info.obj = obj;
info.nnz = sum(abs(x)>1e-10);
if phase2 == 1
    info.Ax = Ax;
    info.Atxi = Atxi;
end
if (printminoryes)
    if ~isempty(msg); fprintf('\n %s',msg); end
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------');
    fprintf('\n  number iter = %2.0d',iter);
    fprintf('\n  time = %3.2f',ttime);
    if iter >= 1; fprintf('\n  time per iter = %5.4f',ttime/iter); end
    fprintf('\n  cputime = %3.2f', ttime_cpu);
    fprintf('\n  primobj = %9.8e, dualobj = %9.8e, relgap = %3.2e',primobj,dualobj, relgap);
    fprintf('\n  primfeas    = %3.2e, dualfeas    = %3.2e',...
        primfeasorg, dualfeasorg);
    if iter >= 1; fprintf('\n  Total CG number = %3.0d, CG per iter = %3.1f', ttCG, ttCG/iter);end
    fprintf('\n  eta = %3.2e, etaorg = %3.2e', eta, etaorg);
    fprintf('\n  min(X)    = %3.2e, max(X)    = %3.2e',...
        info.minx,info.maxx);
    fprintf('\n  number of nonzeros in x (abs(x)>1e-10) = %3d', sum(abs(x)>1e-10));
    fprintf('\n  sigmaorg = %3.2e', sigma*bscale/cscale);
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------\n');
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
elseif (iter < inf)  %% better than (iter < 1000)
    sigma_update_iter = 100;
end
%%**********************************************************************