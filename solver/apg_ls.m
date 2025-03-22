function [primobj,x,info,runhist] = apg_ls(Ainput,b,n,regularizer,lambda1,options,x0)
stoptol = 1e-6;
maxiter = 20000;
printminoryes = 1;
if isfield(options,'maxiter'); maxiter = options.maxiter; end
if isfield(options,'stoptol');     stoptol = options.stoptol; end
%%
tstart = clock;
tstart_cpu = cputime;
if isstruct(Ainput)
    if isfield(Ainput,'A'); A = Ainput.A; end
    if isfield(Ainput,'Amap'); Amap = Ainput.Amap; end
    if isfield(Ainput,'ATmap'); ATmap = Ainput.ATmap; end
else
    A = Ainput;
    Amap =@(x) A*x;
    ATmap = @(y) (y'*A)'; 
end
m = length(b);
ATb = ATmap(b);
%% loss function and its gradient
if (n <= 8000) && (m <= 50000)
    ATA = A'*A;
    ATAmap = @(x) ATA*x;
    gradF = @(x) ATA*x-ATb;
    funF  = @(x) 0.5*(x'*(ATA*x-2*ATb)+b'*b);
else
    ATAmap = @(x) ATmap(Amap(x));
    gradF = @(x) ATmap(Amap(x)-b);
    funF  = @(x) 0.5*norm(Amap(x)-b)^2;
end
%% Proximal mapping and proximal residual function
if strcmp(regularizer.name,'lasso')
    funp = @(x,lambda) lambda*sum(abs(x));
    prox_fun = @(x,lambda) proxL1(x,lambda);
    R_fun = @(x,lambda) x - proxL1(x-gradF(x),lambda);
elseif strcmp(regularizer.name,'exclusive lasso')
    funp = @(x,lambda) lambda1*xgroupnorm(x,options.group_info);
    prox_fun = @(x,lambda) prox_exclusive(x,n,lambda,options.group_info);
    R_fun = @(x,lambda) x - prox_exclusive(x-gradF(x),n,lambda,options.group_info);
elseif strcmp(regularizer.name,'sparse group lasso')
    funp = @(x,lambda) lambda*regularizer.corg(2)*options.P.Lasso_fz(options.P.matrix*(x))+lambda*regularizer.corg(1)*sum(abs(x));
    prox_fun = @(x,lambda) Prox_p(x,lambda*regularizer.corg,options.P);
    R_fun = @(x,lambda) x - Prox_p(x-gradF(x),lambda*regularizer.corg,options.P);
elseif strcmp(regularizer.name,'slope')
    funp = @(x,lambda) lambda*options.lambda_BH*sort(abs(x),'descend');
    prox_fun = @(x,lambda) proxSortedL1(x,lambda*options.lambda_BH');
    R_fun = @(x,lambda) x - proxSortedL1(x-gradF(x),lambda*options.lambda_BH');
end
%%
fprintf('\n-------------------------------------------------')
fprintf('\n An APG method for regularized least squares problems');
fprintf('\n******************************************************');
fprintf('*******************************');
fprintf('\n m = %3.0f, n = %3.0f',m, n);
fprintf('\n-------------------------------------------------')
fprintf('\n  iter | normxdiff |  primobj  |  time  |  eta');
%% 
if ~exist('x0','var')
    x = zeros(n,1);
else
    x = x0;
end
%% Main loop
Lipinv = 1/options.Lip;
breakyes = 0;
Liplam1 = lambda1 * Lipinv;
for iter = 1:maxiter
    if (iter == 1)
        t = 1;
        xtmp = x;
    else
        t  = (1+sqrt(1+4*told^2))/2;
        const = (told-1)/t;
        xtmp  = x + const*(x-xold);
    end
    told = t;
    xold = x;
    if exist('ATA','var')
        ATAxtmp = ATA*xtmp;
    else
        ATAxtmp = ATAmap(xtmp);
    end
    xinput = xtmp - Lipinv*ATAxtmp + Lipinv*ATb;
    x = prox_fun(xinput,Liplam1);
    %% check for termination
    grad = gradF(x);
    R_value = R_fun(x,lambda1);
    eta = norm(R_value) / (1 + norm(grad) + norm(x));
    if  eta < stoptol
        breakyes = 1;
        msg = 'eta: converged';
    end
    if etime(clock, tstart) > 0.5*3600
        breakyes = 777;
        msg = 'time out';
    end
    %%
    if (rem(iter,200)==1) || (breakyes) || (iter==maxiter)
        normx = norm(x);
        normxdiff = norm(x-xold)/max(1,normx);
        primobj = funF(x) + funp(x,lambda1);
        fprintf('\n %5.0f |  %3.2e |  %3.2e  |  %3.2f  | %3.2e ',iter,normxdiff,primobj,etime(clock,tstart),eta);
    end
    ttime = etime(clock,tstart);
    runhist.primobj(iter)   = primobj;
    runhist.time(iter)      = ttime;
    if (breakyes > 0)
        fprintf('\n  breakyes = %3.1f, %s',breakyes,msg);
        break;
    end
end
if (iter == maxiter)
    msg = ' maximum iteration reached';
end
%%
grad = gradF(x);
R_value = R_fun(x,lambda1);
eta = norm(R_value) / (1 + norm(grad) + norm(x));
ttime = etime(clock,tstart);
ttime_cpu = cputime - tstart_cpu;
info.m = m;
info.n = n;
info.Lip = options.Lip;
info.iter = iter;
info.time = ttime;
info.time_cpu = ttime_cpu;
info.eta = eta;
info.etaorg = R_value;
info.obj = primobj;
if (printminoryes)
    if ~isempty(msg); fprintf('\n %s',msg); end
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------');
    fprintf('\n  number iter = %2.0d',iter);
    fprintf('\n  time =  %3.2f',ttime);
    if iter >= 1; fprintf('\n  time per iter = %5.4f',ttime/iter); end
    fprintf('\n  cputime = %3.2f', ttime_cpu);
    fprintf('\n     primobj = %10.9e',primobj);
    fprintf('\n  eta = %3.2e', eta);
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------\n');
end
end