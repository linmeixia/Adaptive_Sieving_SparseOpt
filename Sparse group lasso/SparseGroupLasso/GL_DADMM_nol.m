%% Use ADMM to solve the dual problem of the following 
%% group lasso problem with non-overlapped groups:
%%  min   (1/2)|| A x - b||^2 + p(x),
%% where p(x) = c1 \|x\|_1 + c2 * sum_i w_i ||x_{G_i}||. 
%% (Assumption: P^*P = I)
%%
%% Dual problem:
%% - min (1/2)||y||^2 + <b,y> + P^*(z)
%%   s.t. A^* y + z = 0.
%%
%% June 26,2017 Zhang Yangjing


function [y,z,x,runhist]= GL_DADMM_nol(A,b,c,P,admmopts,y,z,x)

[n,p] = size(A);
c1 = c(1); c2 = c(2);%penalizing parameters
normb = norm(b);
sigold = 0;
%% Initilization
if ~exist('y','var') || ~exist('z','var') || ~exist('x','var')
  y = zeros(n,1); z = zeros(p,1); x = z;
end

%% Parameters
tol = 1e-6;  
printyes = 1;
printminoryes = 1;
sig = 1; %parameter in the Augmented Lagrangian function    
beta = 1.618; %steplength
maxit = 10000;  
stopopt = 2; %1:relgap+feas 2:kkt+gap
stop = 0;
%print_iter = 100;

if n < 3000
    lin_solver = 1; %direct solver
else
    lin_solver = 2; %CG solver
end

N_AAt = 0;
% N_AAt = 0 -- set S=0
% N_AAt = 6 -- take 6 largest eigenvalues of AAt to obtain proximal term S

A_identity = 0;% 1:if A=I
prim_win = 0; dual_win = 0;

if isfield(admmopts,'tol'); tol = admmopts.tol; end
if isfield(admmopts,'tol_gap'); tol_gap = admmopts.tol_gap; end
if isfield(admmopts,'printyes'); printyes = admmopts.printyes; end
if isfield(admmopts,'printminoryes'); printminoryes = admmopts.printminoryes; end
if isfield(admmopts,'Lip'); 
    Lip = admmopts.Lip; 
else
    Lip = svds(A,1)^2;
end
sigLip = 1/Lip;
%sig = max([1/sqrt(Lip),min([1,sigLip,c1]),1e-4]);
sig = max([1/sqrt(Lip),min([1,sigLip,c1,1/c1,1/c2]),1e-8]);
if isfield(admmopts,'beta'); beta = admmopts.beta; end
if isfield(admmopts,'maxit'); maxit = admmopts.maxit; end
if isfield(admmopts,'N_AAt'); N_AAt = admmopts.N_AAt; end
if isfield(admmopts,'A_identity');A_identity = admmopts.A_identity; end
if isfield(admmopts,'lin_solver');lin_solver = admmopts.lin_solver; end
if isfield(admmopts,'stopopt');stopopt = admmopts.stopopt; end

if (stopopt == 3) && ~exist('tol_gap','var')
    fprintf('Pleas input the tolerance for  relative gap');
end
tstart = clock;

%% proximal term S 
if isfield(admmopts,'AAt')
    AAt = admmopts.AAt;
else
    AAt = A*A';
end
AAt0 = AAt;
S = zeros(n,n);
if N_AAt > 0
    l = N_AAt;
    [V,D] = eigs(AAt,l);
    dd = diag(D);
    lam_A = dd(end);

    if (D(l - 1,l - 1) <= lam_A )
        fprintf('\nError!AAt:(lambda(%d)=%d)is smaller than (lambda(%d)=%d)',...
            l - 1,D(l - 1,l - 1),l,lam_A);
    else
        fprintf('\nThe first %d eigenvalues of AAt:\n',l);
        g = sprintf('%6.3f| ',dd);
        fprintf(' %s\n',g);
    end
    V = V(:,1:l-1);
    dd = dd(1:end-1);
    S = V*diag(dd - lam_A)*V';
    S = S - AAt + lam_A*speye(n);
end


%%
fprintf('\n ***********************************************');
fprintf('******************************************');
fprintf('\n \t\t GROUP LASSO: ADMM      ');
fprintf('\n *********************************************');
fprintf('*******************************************\n');
fprintf('\nm=%d, n=%d',n,p);
fprintf('\ntol=%1.1e, reg_para:c1=%4.3f, c2=%4.3f\n',tol,c1,c2);
if printminoryes    
    fprintf('\n it   pobj        dobj         pinf        dinf       sigma')
    fprintf('     eta1     rel_gap   inf_ratio\n')
end

eta_1 = 0;
%% main loop
    
for iter = 1:maxit
    %update y
    rhs = z - x/sig ;
    rhs = A*rhs;
    rhs = - (b/sig + rhs); 
    
    if N_AAt > 0        
        rhs = rhs + S*y;
        lam_Asig = lam_A + 1/(sig);
        ddinv = dd + 1/(sig);
        ddinv = 1./ddinv;
        ddinv = ddinv - 1/lam_Asig;
        y = V'*rhs;
        y = diag(ddinv)*y;
        y = V*y;
        y = rhs/lam_Asig + y; 
    else
        if A_identity == 1
            y = sig*rhs/(1+sig);
            cg_res1 = 0; cg_iter1 = 0;
        else
            if lin_solver == 1
                if (sigold ~= sig) || iter == 1
                    for i = 1:n
                        AAt0(i,i) = AAt(i,i) + 1/sig;
                    end
                    LAAt = mycholAAt(AAt0,n);
                end
                y = mylinsysolve(LAAt,rhs);
            elseif lin_solver == 2
                %cg_tol = 1/(iter + 1)^2*(1e-4);            
                cg_tol = max(min(5e-2,1/(iter^3)),1e-10);            
                cg_maxit = 50;  
                if (sigold ~= sig) || iter == 1
                    for i = 1:n
                        AAt0(i,i) = AAt(i,i) + 1/sig;
                    end
                end               
                [y,~,cg_res1,cg_iter1] = ...
                pcg(AAt0,rhs,cg_tol,cg_maxit,[],[],y);  
            end
        end
    end
 
    
    %update z
    Aty = A'*y;
    Atyx = x/sig - Aty;
    z = Atyx - Prox_p(Atyx,c,P);
        
    %update multipliers x
    Rd = Aty + z;
    x = x - beta*sig*Rd;
    
    
    %% compute things: pobj,dobj,infeas,gap,...
    normz = norm(z);   
    Ax = A*x; Px = P.matrix*(x);
    dualfeas = norm(Rd)/(1+normz);
    primfeas = norm(y - (Ax - b))/(1+normb);    
    lasso = c2*P.Lasso_fz(Px) + c1*sum(abs(x));
    dualobj = -sum(y.*y)/2 - b'*y;
    primobj = sum((Ax - b).^2)/2 + lasso;

    gap = primobj - dualobj;
    relgap = abs(gap)/(1+abs(primobj)+abs(dualobj));    
    
    if stopopt == 1
        stop = max(dualfeas,relgap) < tol;
    elseif stopopt == 2
        eta_tmp = max([dualfeas,primfeas]);%max([relgap,dualfeas,primfeas]);
        if eta_tmp < tol
            grad = A'*(A*x-b);
            tmp2 = x-Prox_p(x-grad,c,P);   
            eta_1 = norm(tmp2)/(1+norm(grad)+norm(x));
%             eta_1 = norm(x - Prox_p(x+z,c,P))/(1+norm(x));
            stop = eta_1 < tol;
        end
    elseif stopopt == 3
        stop = (dualfeas < tol) && (relgap < tol_gap);
    elseif stopopt ==4
        stop = (dualfeas < tol) && (gap < tol*norm(b,2)^2);
    end
    
    
%    dualfeas = norm(Rd)/(1+normz);%norm(Rd)/(1+norm(Aty)+normz);
%     primfeas = norm(y - (Ax - b))/(1+normb);%norm(y - (Ax - b))/(1+norm(Ax)+normb);
    
%     lasso = c2*P.Lasso_fz(Px) + c1*sum(abs(x));
%     dualobj = -sum(y.*y)/2 - b'*y;
%     primobj = sum((Ax - b).^2)/2 + lasso;
%     
%     gap = primobj - dualobj;
%     relgap = abs(gap)/(1+abs(primobj)+abs(dualobj));
%     
%     eta_1 = norm(x - Prox_p(x+z,c,P))/(1+norm(x));
% 
%     stop = max([eta_1,primfeas,dualfeas,relgap]) < tol;

%     stop = 0;
%     eta_tmp = max([dualfeas,relgap]);
%     if eta_tmp < tol
%         eta_1 = norm(x - Prox_p(x+z,c,P))/(1+norm(x));
%         stop = max([eta_1,eta_tmp]) < tol;
%     end
        

    ttime = etime(clock,tstart);
    
    runhist.primfeas(iter)    = primfeas; 
    runhist.dualfeas(iter)    = dualfeas; 
    runhist.sigma_seq(iter)   = sig; 
    runhist.primobj(iter)     = primobj;
    runhist.dualobj(iter)     = dualobj;
    runhist.gap(iter)         = gap;
    runhist.relgap(iter)      = relgap; 
    runhist.ttime(iter)       = ttime;
    
    if (iter <= 20)
        print_iter = 5; %20
    elseif (iter <= 200)
        print_iter = 50;
    elseif (iter <= 3000)
        print_iter = 500; 
    else
        print_iter = 2000; 
    end
 
    if printminoryes && mod(iter,print_iter) == 0 && (N_AAt == 0) && (lin_solver == 2)
        fprintf('\n----[step1 |CG_iter:%d |res:%2.2e]',cg_iter1,cg_res1);
        fprintf('\n');
    end          
    if printminoryes && mod(iter,print_iter) == 0
        fprintf('%d  ', iter);     
        fprintf('%4.4e  ',primobj);
        fprintf('%4.4e  ',dualobj);
        fprintf('%4.4e  ',primfeas);
        fprintf('%4.4e  ',dualfeas);
        fprintf('%2.2e  ',sig);
        fprintf('%4.1e  ',eta_1);
        fprintf('%2.2e  ',relgap);         
    end
 
    
    %% check termination
    if iter == maxit || stop
        admm_termination = 'converged';
        if iter == maxit, admm_termination = 'maxiter reached'; end
        runhist.termination = admm_termination;
        
        runhist.iter = iter;
%         runhist.nnz = length(find(abs(x)>tol));
        runhist.nnz = cardcal(x,0.999);
        if stopopt == 2
            runhist.kktres = max(eta_tmp,eta_1);
        end
        if printyes
            fprintf('\n****************************************\n');
            fprintf([' ADMM        : ',admm_termination,'\n']);
            fprintf(' iteration   : %d\n',iter);
            
            fprintf(' dual_obj    : %4.8e\n',runhist.dualobj(iter));
            fprintf(' prim_obj    : %4.8e\n',runhist.primobj(iter));
            fprintf(' relgap      : %4.5e\n',runhist.relgap(iter));
            fprintf(' nnz         : %d\n',runhist.nnz);

%             fprintf('\n\n');
        end
        
        break;
    end
    
    %% update sigma
    ratio = primfeas / (dualfeas + eps);
    runhist.ratio_seq(iter) = ratio;
    if printminoryes && mod(iter,print_iter) == 0
        
        fprintf('%2.2f  ',runhist.ratio_seq(iter));
        fprintf('\n');
    end
    if ratio < 1%1 for pathway
        prim_win = prim_win + 1;
    else
        dual_win = dual_win + 1;
    end
    sigma_update_iter = sigma_fun(iter); 
    sigmascale = 1.25;%1.25
    sigold = sig;
    if (rem(iter,sigma_update_iter)==0) 
        sigmamax = 1e6; sigmamin = 1e-8;
        if iter <= 2500
            if (prim_win > max(1,1.2*dual_win)) 
                prim_win = 0;
                sig = min(sigmamax,sig*sigmascale);
            elseif(dual_win > max(1,1.2*prim_win))
                dual_win = 0;
                sig = max(sigmamin,sig/sigmascale);
            end
        else
            ratiosub = runhist.ratio_seq([max(1,iter-19):iter]);  
            meanratiosub = mean(ratiosub);
            if meanratiosub < 0.1 || meanratiosub > 1/0.1
               sigmascale = 1.4; 
            elseif meanratiosub < 0.2 || meanratiosub > 1/0.2
               sigmascale = 1.35;
            elseif meanratiosub < 0.3 || meanratiosub > 1/0.3
               sigmascale = 1.32;
            elseif meanratiosub < 0.4 || meanratiosub > 1/0.4
               sigmascale = 1.28;
            elseif meanratiosub < 0.5 || meanratiosub > 1/0.5
               sigmascale = 1.26;
            end
            primidx = find(ratiosub <= 1);
            dualidx = find(ratiosub >  1);
            if (length(primidx) >= 12) 
               sig = min(sigmamax,sig*sigmascale);
            end
            if (length(dualidx) >= 12)
               sig = max(sigmamin,sig/sigmascale); 
            end 
        end
    end
    
end%end iter
end%end ADMM

%%*************************************************************************
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
  else
     sigma_update_iter = 100;
  end
  
end
%%*************************************************************************
 function q = mylinsysolve(L,r) 
    if strcmp(L.matfct_options,'chol')
       q(L.perm,1) = mextriang(L.R, mextriang(L.R,r(L.perm),2) ,1);
    elseif strcmp(L.matfct_options,'spcholmatlab')
       q(L.perm,1) = mexbwsolve(L.Rt,mexfwsolve(L.R,r(L.perm,1)));
    end
 end
 %%*************************************************************************