%%*************************************************************************
%% SSNAL:
%% Copyright (c) 2019 by
%% Meixia Lin, Defeng Sun, Kim-Chuan Toh, Yancheng Yuan 
%%*************************************************************************
function [u,w,Atxi,xi,par,runhist,info] = ...
    ppa_sub_logistic_lasso(b,Ainput,x0,Ax0,Atxi0,xi0,lambda1,par,options)
iftest = 0;
printsub = 1;

breakyes = 0;
maxitersub = 50;
tol = 1e-6;
maxitpsqmr =500;

if isfield(options,'iftest'); iftest = options.iftest; end
if isfield(options,'printsub'); printsub = options.printsub; end
if isfield(options,'maxitersub'); maxitersub = options.maxitersub; end
if isfield(options,'tol'); tol = min(tol,options.tol); end

sig = par.sigma;
sigdtau = par.sigdtau;
bscale = options.bscale;
cscale = options.cscale;
n = length(x0);
%% preperation
Amap = @(x) Ainput.Amap(x);
ATmap = @(x) Ainput.ATmap(x);
par.lsAmap = Amap;
par.lambda1 = lambda1;
par.b = b;

uinput = x0 - sig*Atxi0;
[up,info_u] = proxL1(uinput,sig*lambda1); 
par.info_u = info_u;
u = (uinput -up)/sig;

winput = sigdtau*xi0 + Ax0;
[wp,info_w] = proximal_mapping_logistic(winput,b,sigdtau);
par.info_w = info_w;
w = (winput-wp)/sigdtau;

Atxi = Atxi0; xi = xi0;
Ly = - norm(uinput)^2/(2*sig) + (sig/2)*norm(u)^2 - norm(winput)^2/(2*sigdtau) + (sigdtau/2)*norm(w)^2;
Ly = Ly + lambda1*norm(up,1) + sum(log(1+exp(-b.*wp)));
runhist.psqmr(1) = 0;
runhist.findstep(1) = 0;

const_Ly = norm(x0)^2/(2*sig) + norm(Ax0)^2/(2*sigdtau);
%% main Newton iteration
for itersub = 1:maxitersub
    Rd1 = Atxi + u;
    normRd1 = norm(Rd1);
    Rd2 = w - xi;
    normRd2 = norm(Rd2);
    Aup = Amap(up);
    GradLxi = -(wp - Aup);
    normGradLxi = norm(GradLxi)*sqrt(bscale*cscale)/(1+norm(xi)*sqrt(bscale*cscale));
    priminf_sub = normGradLxi;
    dualinf_sub = max(normRd1*cscale/(1+norm(u)*cscale),normRd2*sqrt(bscale*cscale)/(1+norm(xi)*sqrt(bscale*cscale)));
    if max(priminf_sub,dualinf_sub) < tol
        tolsubconst = 0.01;%0.1;
    else
        tolsubconst = 0.005;%0.05; 
    end
    tolsub = max(min(1,par.tolconst*dualinf_sub),tolsubconst*tol);
    runhist.priminf(itersub) = priminf_sub;
    runhist.dualinf(itersub) = dualinf_sub;
    runhist.Ly(itersub)      = Ly;
    if (printsub)
        fprintf('\n      %2.0d  %- 11.10e [%3.2e %3.2e]',...
            itersub,Ly,priminf_sub,dualinf_sub);
    end
    psix = sum(log(1+exp(-b.*Aup))) + lambda1*norm(up,1) + norm(up-x0)^2/(2*sig) + norm(Aup-Ax0)^2/(2*sigdtau);
%     tolsub = 1e-8;
    if abs(psix-Ly-const_Ly) < max(tolsub) && itersub > 1
        msg = 'good termination in subproblem:';
        if printsub
            fprintf('\n       %s  ',msg);
            fprintf(' normRd1=%3.2e, normRd2=%3.2e, gap_sub=%3.2e, gradLxi = %3.2e, tolsub=%3.2e',...
                normRd1,normRd2,abs(psix-Ly-const_Ly),normGradLxi,tolsub);
        end
        breakyes = -1;
        break;
    end
    %% Compute Newton direction
    %% precond = 0,
    par.epsilon = min([1e-3,0.1*normGradLxi]); %% good to add
    if (dualinf_sub > 1e-3) || (itersub <= 5)
        maxitpsqmr = max(maxitpsqmr,200);
    elseif (dualinf_sub > 1e-4)
        maxitpsqmr = max(maxitpsqmr,300);
    elseif (dualinf_sub > 1e-5)
        maxitpsqmr = max(maxitpsqmr,400);
    elseif (dualinf_sub > 5e-6)
        maxitpsqmr = max(maxitpsqmr,500);
    end
    if (itersub > 1)
        prim_ratio = priminf_sub/runhist.priminf(itersub-1);
        dual_ratio = dualinf_sub/runhist.dualinf(itersub-1);
    else
        prim_ratio = 0; dual_ratio = 0;
    end
    rhs = GradLxi;
    tolpsqmr = min([5e-3, 0.1*norm(rhs)]);
    const2 = 1;
    if itersub > 1 && (prim_ratio > 0.5 || priminf_sub > 0.1*runhist.priminf(1))
        const2 = 0.5*const2;
    end
    if (dual_ratio > 1.1); const2 = 0.5*const2; end
    tolpsqmr = const2*tolpsqmr;
    par.tol = tolpsqmr; par.maxit = 2*maxitpsqmr;
     
    par.xiold = xi;
    [dxi,resnrm,solve_ok,par] = ppa_Netwonsolve_logistic_lasso(Ainput,rhs,par);
    
    
    Atdxi = ATmap(dxi);
    iterpsqmr = length(resnrm)-1;
    
    if (iftest)
        testd = test_Jacobian_Exclusivelasso(Ainput,n,b,-GradLxi,xi,Atxi,x0,sig,dxi,Atdxi,lambda1,group_info,par);
        fprintf('\n test_direction normd=%3.2e %3.2e %3.2e %3.2e %3.2e %3.2e %3.2e %3.2e %3.2e %3.2e %3.2e %3.2e %3.2e\n',norm(dxi),testd);
    end
    
    if (printsub)
        if par.innerop == 1
            fprintf('| [%3.1e %3.1e %3.1d]',par.tol,resnrm(end),iterpsqmr);
        end
    end
    par.iter = itersub;
    if (itersub <= 3) && (dualinf_sub > 1e-4) || (par.iter <3)
        stepop = 1;
    else
        stepop = 2;
    end
    steptol = 1e-5;
    step_op.stepop = stepop;
    step_op.Amap = Amap;
    [par,Ly,xi,Atxi,u,up,w,wp,alp,iterstep] = ...
        findstep(par,n,b,lambda1,Ly,xi,Atxi,...
        u,up,w,wp,dxi,Atdxi,steptol,step_op);
    runhist.solve_ok(itersub) = solve_ok;
    runhist.psqmr(itersub)    = iterpsqmr;
    runhist.findstep(itersub) = iterstep;
    Ly_ratio = 1;
    if (itersub > 1)
        Ly_ratio = (Ly-runhist.Ly(itersub-1))/(abs(Ly)+eps);
    end
    if (printsub)
        fprintf(' | %3.2e %2.0f',alp,iterstep);
        if (Ly_ratio < 0); fprintf('-'); end
    end
    %% check for stagnation
    if (itersub > 4)
        idx = max(1,itersub-3):itersub;
        tmp = runhist.priminf(idx);
        ratio = min(tmp)/max(tmp);
        if (all(runhist.solve_ok(idx) <= -1)) && (ratio > 0.9) ...
                && (min(runhist.psqmr(idx)) == max(runhist.psqmr(idx))) ...
                && (max(tmp) < 5*tol)
            fprintf('#')
            breakyes = 1;
        end
        const3 = 0.7;
        priminf_1half  = min(runhist.priminf(1:ceil(itersub*const3)));
        priminf_2half  = min(runhist.priminf(ceil(itersub*const3)+1:itersub));
        priminf_best   = min(runhist.priminf(1:itersub-1));
        priminf_ratio  = runhist.priminf(itersub)/runhist.priminf(itersub-1);
        stagnate_idx   = find(runhist.solve_ok(1:itersub) <= -1);
        stagnate_count = length(stagnate_idx);
        idx2 = max(1,itersub-7):itersub;
        if (itersub >= 10) && all(runhist.solve_ok(idx2) == -1) ...
                && (priminf_best < 1e-2) && (dualinf_sub < 1e-3)
            tmp = runhist.priminf(idx2);
            ratio = min(tmp)/max(tmp);
            if (ratio > 0.5)
                if (printsub); fprintf('##'); end
                breakyes = 2;
            end
        end
        if (itersub >= 15) && (priminf_1half < min(2e-3,priminf_2half)) ...
                && (dualinf_sub < 0.8*runhist.dualinf(1)) && (dualinf_sub < 1e-3) ...
                && (stagnate_count >= 3)
            if (printsub); fprintf('###'); end
            breakyes = 3;
        end
        if (itersub >= 15) && (priminf_ratio < 0.1) ...
                && (priminf_sub < 0.8*priminf_1half) ...
                && (dualinf_sub < min(1e-3,2*priminf_sub)) ...
                && ((priminf_sub < 2e-3) || (dualinf_sub < 1e-5 && priminf_sub < 5e-3)) ...
                && (stagnate_count >= 3)
            if (printsub); fprintf(' $$'); end
            breakyes = 4;
        end
        if (itersub >=10) && (dualinf_sub > 5*min(runhist.dualinf)) ...
                && (priminf_sub > 2*min(runhist.priminf)) %% add: 08-Apr-2008
            if (printsub); fprintf('$$$'); end
            breakyes = 5;
        end
        if (itersub >= 20)
            dualinf_ratioall = runhist.dualinf(2:itersub)./runhist.dualinf(1:itersub-1);
            idx = find(dualinf_ratioall > 1);
            if (length(idx) >= 3)
                dualinf_increment = mean(dualinf_ratioall(idx));
                if (dualinf_increment > 1.25)
                    if (printsub); fprintf('^^'); end
                    breakyes = 6;
                end
            end
        end
        if breakyes > 0
            Rd =  Atxi + u;
            normRd = norm(Rd);
            Aup = Amap(up);
            fprintf('\n new dualfeasorg = %3.2e', normRd*cscale/(1+norm(u)*cscale));
            break
        end
    end
end
info.breakyes = breakyes;
info.itersub = itersub;
info.tolconst = par.tolconst;
info.up = up;
info.wp = wp;
info.Aup = Aup;
info.innerop = par.innerop;
end

function [par,Ly,xi,Atxi,u,up,w,wp,alp,iter] = ...
    findstep(par,n,b,lambda1,Ly0,xi0,Atxi0,...
    u0,up0,w0,wp0,dxi,Atdxi,tol,options)
printlevel = 0;
if isfield(options,'stepop'); stepop = options.stepop; end
if isfield(options,'printlevel'); printlevel = options.printlevel; end
maxit = ceil(log(1/(tol+eps))/log(2));
c1 = 1e-4; c2 = 0.9;
sig = par.sigma;
sigdtau = par.sigdtau;
%%
g0  = dxi'* (- wp0) + Atdxi'*up0;
if  (g0 <= 0) %(g0 < 1e-10*norm(grad0))
    alp = 0; iter = 0;
    if (printlevel)
        fprintf('\n Need an ascent direction, %2.1e  ',g0);
    end
    xi = xi0;
    Atxi = Atxi0;
    u = u0;
    up = up0;
    w = w0;
    wp = wp0;
    Ly = Ly0;
    return;
end
%%
alp = 1; alpconst = 0.5;
for iter = 1:maxit
    if (iter == 1)
        alp = 1; LB = 0; UB = 1;
    else
        alp = alpconst*(LB+UB);
    end
    xi = xi0 + alp*dxi;
    
    uinput = up0 + sig*u0 - sig*alp*Atdxi;
    [up,info_u] = proxL1(uinput,sig*lambda1); 
    par.info_u = info_u;
    u = (uinput -up)/sig;
    
    winput = sigdtau*w0 + wp0 + sigdtau*alp*dxi;
    [wp,info_w] = proximal_mapping_logistic(winput,b,sigdtau);
    par.info_w = info_w;
    w = (winput-wp)/sigdtau;
   
    galp = dxi'*(- wp) + Atdxi'*up; 
    Ly = - norm(uinput)^2/(2*sig) + (sig/2)*norm(u)^2 - norm(winput)^2/(2*sigdtau) + (sigdtau/2)*norm(w)^2;
    Ly = Ly + lambda1*norm(up,1) + sum(log(1+exp(-b.*wp)));
    if printlevel
        fprintf('\n ------------------------------------- \n');
        fprintf('\n alp = %4.3f, LQ = %11.10e, LQ0 = %11.10e',alp,Ly,Ly0);
        fprintf('\n galp = %4.3f, g0 = %4.3f',galp,g0);
        fprintf('\n ------------------------------------- \n');
    end
    if (iter==1)
        gLB = g0; gUB = galp;
        if (sign(gLB)*sign(gUB) > 0)
            if (printlevel); fprintf('|'); end
            Atxi = Atxi0+alp*Atdxi;
            return;
        end
    end
    if ((abs(galp) < c2*abs(g0))) && (Ly-Ly0-c1*alp*g0 > -1e-12/max(1,abs(Ly0)))
        if (stepop==1) || ((stepop == 2) && (abs(galp) < tol))
            if (printlevel); fprintf(':'); end
            Atxi = Atxi0+alp*Atdxi;
            return
        end
    end
    if (sign(galp)*sign(gUB) < 0)
        LB = alp; gLB = galp;
    elseif (sign(galp)*sign(gLB) < 0)
        UB = alp; gUB = galp;
    end
end
if iter == maxit
    Atxi = Atxi0+alp*Atdxi;
end
if (printlevel); fprintf('m'); end
end
% %%********************************************************************