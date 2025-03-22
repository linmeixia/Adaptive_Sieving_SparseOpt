%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Use the ADMM to solve the SLOPE model:
%% min { 0.5*||Ax - b||^2 + kappa_lambda (x) 
%% from the dual pespective: 
%% - min {0.5*||xi||^2 + b'*xi + kappa^*_lambda (u): A'*xi + u = 0}
%% where kappa_lambda(x)=lambda'*sort(abs(x),'descend')  
%% and kappa^*_lambda is the conjugate function, [m,n]=size(A) with m<n
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   function [obj,xi,u,x,info,runhist] = ADMM(Ainput,b,n,lambda1,options,x0,xi0,u0)
   rng('default');           
   maxiter       = 20000;  
   sigma         = 1;
   gamma         = 1.618;
   stoptol       = 1e-6;
   printyes      = 1;
   printminoryes = 1;
   sig_fix       = 0;
   rescale       = 1;
   use_infeasorg = 0;
   phase2        = 0;  
   Asolver       = 'prox';
   stopop        = 2;
   gapcon        = 1;
   pdconst       = 1;
   if isfield(options,'sigma');         sigma         = options.sigma;         end
   if isfield(options,'gamma');         gamma         = options.gamma;         end
   if isfield(options,'stoptol');       stoptol       = options.stoptol;       end
   if isfield(options,'printyes');      printyes      = options.printyes;      end
   if isfield(options,'maxiter');       maxiter       = options.maxiter;       end
   if isfield(options,'printminoryes'); printminoryes = options.printminoryes; end
   if isfield(options,'sig_fix');       sig_fix       = options.sig_fix;       end
   if isfield(options,'dscale');        dscale        = options.dscale;        end
   if isfield(options,'rescale');       rescale       = options.rescale;       end
   if isfield(options,'use_infeasorg'); use_infeasorg = options.use_infeasorg; end
   if isfield(options,'phase2');        phase2        = options.phase2;        end
   if isfield(options,'sGS');           sGS           = options.sGS;           end
   if isfield(options,'Asolver');       Asolver       = options.Asolver;       end
%% Amap and ATmap
   tstart     = clock;
   tstart_cpu = cputime;
   m          = length(b); 
   if isstruct(Ainput)
      if isfield(Ainput,'A');     A0     = Ainput.A;     end
      if isfield(Ainput,'Amap');  Amap0  = Ainput.Amap;  end
      if isfield(Ainput,'ATmap'); ATmap0 = Ainput.ATmap; end
   else
      A0     = Ainput; 
      Amap0  = @(x) A0*x;
      ATmap0 = @(y) A0'*y;
   end
   AATmap0   = @(x) Amap0(ATmap0(x));
   Amap      = Amap0;
   ATmap     = ATmap0;
   AATmap    = AATmap0;
   if strcmp(Asolver,'prox')
      eigsopt.issym = 1;
      rA            = 1;
      [VA,dA,~]     = eigs(AATmap0,m,rA,'LA',eigsopt);
      dA            = diag(dA); 
      rA            = sum(dA>0);
      for i= 1:rA
         fprintf('\n %d th eigen = %3.2e',i, dA(i));
      end
      proxA   = min(10,rA);
      dA      = dA(1:proxA);
      VA      = VA(:,1:proxA);
      VAt     = VA';
      MAmap   = @(xi) dA(end)*xi + VA*((dA-dA(end)).*(VAt*xi));
      MAinv   = @(xi,sigma)...
                xi/(1+sigma*dA(end)) + VA(:,1:proxA)*((1./(1 + sigma*dA(1:proxA)) - 1/(1 + sigma*dA(end))).*(VAt(1:proxA,:)*xi));
      pdconst = 5;
   elseif strcmp(Asolver,'direct') && exist('A0','var')
      AAt0    = A0*A0';
   end


%% initiallization
   borg       = b; 
   lambda1org = lambda1;
   normborg   = 1 + norm(borg);
   normb      = normborg;
   if ~exist('x0','var') || ~exist('xi0','var') || ~exist('u0','var') 
      x = zeros(n,1); xi = zeros(m,1); u = zeros(n,1);
   else
      x = x0; xi = xi0; u = u0;
   end
   bscale   = 1; 
   cscale   = 1;
   objscale = bscale*cscale;
   if printyes
      fprintf('\n *******************************************************');
      fprintf('******************************************');
      fprintf('\n \t\t  Phase I:  ADMM  for solving SLOPE with  gamma = %6.3f', gamma);
      fprintf('\n ******************************************************');
      fprintf('*******************************************\n');
      if printminoryes
         fprintf('\n problem size: n = %3.0f, nb = %3.0f',n, m);
         fprintf('\n ---------------------------------------------------');
      end
      fprintf('\n  iter|  [pinfeas  dinfeas] [pinforg dinforg]   relgap |    pobj       dobj      |');
      fprintf(' time |  sigma  |gamma |');
   end

%%
   Atxi   = ATmap(xi); 
   AAtxi  = Amap(Atxi);
   if strcmp(Asolver,'cg')
      IpsigAAtxi = xi + sigma*AAtxi;
   elseif exist('AAt0','var')
      AAt = AAt0;
      Lxi = mychol(eye(m) + sigma*AAt,m);
   end
   Ax     = Amap(x); 
   Rp1    = Ax - b;
   Rd     = Atxi + u;
   ARd    = Amap(Rd);
   
   primfeas    = norm(Rp1 - xi)/normborg;
   dualfeas    = norm(Rd)/(1 + norm(u));
   maxfeas     = max(primfeas, dualfeas);
   primfeasorg = primfeas;
   dualfeasorg = dualfeas;
   maxfeasorg  = maxfeas;
   
   runhist.cputime(1)     = etime(clock,tstart);
   runhist.psqmrxiiter(1) = 0;
   if printyes
      fprintf('\n initial primfeasorg = %3.2e, dualfeasorg = %3.2e', primfeasorg, dualfeasorg);
   end

%% main Loop
   breakyes = 0;
   prim_win = 0;
   dual_win = 0;
   repeaty  = 0;
   msg      = [];
%%
   for iter = 1:maxiter
      if (rescale >= 3 && rem(iter, 203)) == 0 || iter == 1
         normAtxi = norm(Atxi);
         normx    = norm(x); normu = norm(u);
         normuxi  = max([normAtxi,normu]);
      end
      if (((rescale == 1) && (maxfeas < 5e2) && (iter > 21) && (abs(relgap) < 0.5)) ...
         || ((rescale==2) && (maxfeas <1e-2) && (abs(relgap) < 0.05) && (iter > 40)) ...
         || ((rescale>=3) && (max(normx/normuxi,normuxi/normx) > 1.2) && (rem(iter,203)==0)))

         if (rescale <= 2) 
            normAtxi = norm(Atxi);
            normx    = norm(x); 
            normu    = norm(u);
            normuxi  = max([normAtxi,normu]);
         end
         const   = 1;
         bscale2 = normx*const;
         cscale2 = normuxi*const;
         sbc     = sqrt(bscale2*cscale2);
         b       = b/sbc; 
         u       = u/cscale2; 
         lambda1 = lambda1/cscale2;
         x       = x/bscale2; 
         xi      = xi/sbc; 
         Rp1     = Rp1/sbc;
         Amap    = @(x) Amap(x*sqrt(bscale2/cscale2));
         ATmap   = @(x) ATmap(x*sqrt(bscale2/cscale2));
         AATmap  = @(x) AATmap(x*(bscale2/cscale2));
         Ax      = Ax/sbc;
         ARd     = ARd*sqrt(bscale2/cscale2^3);
         if exist('AAt','var');   AAt        = (bscale2/cscale2)*AAt; end                  
         if strcmp(Asolver,'cg'); IpsigAAtxi = IpsigAAtxi/sbc;        end
         if strcmp(Asolver,'prox')
            dA    = dA*bscale2/cscale2;
            MAmap = @(xi) dA(end)*xi + VA*((dA-dA(end)).*(VAt*xi));
            MAinv = @(xi,sigma)...
                  xi/(1+sigma*dA(end)) + VA(:,1:proxA)*((1./(1 + sigma*dA(1:proxA)) - 1/(1 + sigma*dA(end))).*(VAt(1:proxA,:)*xi));
         end
         sigma    = sigma*(cscale2/bscale2);
         cscale   = cscale*cscale2;
         bscale   = bscale*bscale2;
         objscale = objscale*(cscale2*bscale2);
         normb    = 1+norm(b);
         if printyes
            fprintf('\n    ');
            fprintf('[rescale=%1.0f: %2.0f| [%3.2e %3.2e %3.2e] | %3.2e %3.2e| %3.2e]',...
            rescale,iter,normx,normAtxi,normu,bscale,cscale,sigma);
         end         
         rescale  = rescale+1; 
         prim_win = 0; 
         dual_win = 0; 
      end
      xiold      = xi; 
      uold       = u;
      xold       = x; 
      Axold      = Ax;
      %% compute xi

      if strcmp(Asolver,'cg')
         rhsxi       = Rp1 - sigma*(ARd - (IpsigAAtxi - xi)/sigma);
         parxi.tol   = max(0.9*stoptol,min(1/iter^1.1,0.9*maxfeas));
         parxi.sigma = sigma;
         [xi,IpsigAAtxi,resnrmxi,solve_okxi] = psqmry('matvecxi',AATmap,rhsxi,parxi,xi,IpsigAAtxi);
      elseif strcmp(Asolver,'prox')
         rhsxi = Rp1 - sigma*(ARd - MAmap(xi));
         xi    = MAinv(rhsxi,sigma);
      elseif strcmp(Asolver,'direct')
         rhsxi = Rp1 - sigma*Amap(u);
         if m <=300
            xi = (eye(m) + sigma*AAt)\rhsxi;
         else
            xi = mylinsysolve(Lxi,rhsxi);
         end
      end
      Atxi   = ATmap(xi);
      uinput = x - sigma*Atxi;
      up     = proxSortedL1(uinput,sigma*lambda1);        
      u      = (uinput - up)/sigma;
    
      %% update mutilplier Xi, y
      Rd          = Atxi + u;
      x           = xold - gamma*sigma*Rd;
      ARd         = Amap(Rd);
      Ax          = Axold - gamma*sigma*ARd;
      Rp1         = Ax - b;
      normRp      = norm(Rp1 - xi);
      normRd      = norm(Rd);
      normu       = norm(u); 
      etaC        = 0;  
      etaCorg     = 0;  
      primfeas    = normRp/normb;
      dualfeas    = max([normRd/(1+normu),1*etaC]);
      maxfeas     = max(primfeas,dualfeas);
      dualfeasorg = max([normRd*cscale/(1+normu*cscale),1*etaCorg]);
      primfeasorg = sqrt(bscale*cscale)*normRp/normborg;
      maxfeasorg  = max(primfeasorg, dualfeasorg);
      %%-------------------------------------------------------      
      %% record history
      runhist.dualfeas(iter)    = dualfeas;
      runhist.primfeas(iter)    = primfeas;
      runhist.dualfeasorg(iter) = dualfeasorg;
      runhist.primfeasorg(iter) = primfeasorg;
      runhist.maxfeasorg(iter)  = maxfeasorg;
      runhist.sigma(iter)       = sigma;
      if strcmp(Asolver,'cg'); runhist.psqmrxiiter(iter) = length(resnrmxi) - 1; end 
      runhist.xr(iter)          = sum(abs(x)>1e-10); 
%% check for termination
%%--------------------------------------------------------- 
      if stopop == 1
         if (max([primfeasorg,dualfeasorg]) < 5*stoptol) 
            grad   = ATmap0(Rp1*sqrt(bscale*cscale)); 
            etaorg = norm(x*bscale- proxSortedL1(x*bscale - grad,lambda1org));
            eta    = etaorg / (1 + norm(grad) + norm(x*bscale));
            if  eta < stoptol
               breakyes = 1;
               msg      = 'Converged';
            end
            primobj = objscale*(0.5*norm(xi)^2 + lambda1'* sort(abs(x),'descend'));
             
            if max([primfeasorg,dualfeasorg]) < stoptol 
              dualobj      = objscale*(-0.5*norm(xi)^2 -b'*xi + lambda1'*sort(abs(up),'descend')- up'*u);               
               relgap      = abs(primobj-dualobj)/max( 1,abs(primobj)); 
               if abs(relgap) < stoptol && eta < sqrt(stoptol)
                  breakyes = 2;
                  msg      = 'Converged';
               end
            else
               if isfield(options,'optval')
                  if primobj < options.optval
                     breakyes = 3;
                     msg      = 'Opt_value converged';
                  end
               end
            end
         end
      elseif stopop == 2
         if max([primfeasorg,dualfeasorg]) < stoptol 
            primobj = objscale*(0.5*norm(xi)^2 + lambda1'* sort(abs(x),'descend')); 
            dualobj = objscale*(-0.5*norm(xi)^2 -b'*xi + lambda1'*sort(abs(up),'descend')- up'*u);
            relgap  = abs(primobj-dualobj)/max( 1,abs(primobj)); 
           if abs(relgap) < gapcon*stoptol 
               grad   = ATmap0(Rp1*sqrt(bscale*cscale)); 
               etaorg = norm(x*bscale- proxSortedL1(x*bscale - grad,lambda1org));               
               eta    = etaorg / (1 + norm(grad) + norm(x*bscale));
               if eta < stoptol%max(0.01,sqrt(stoptol))
                  breakyes = 88;
                  msg      = 'Converged';
               end
            end
         end
      elseif stopop == 0
         if (max([primfeasorg,dualfeasorg]) < pdconst*stoptol) 
            if ~exist('eta','var') || rem(iter,50) == 1 || eta < 1.2*stoptol
               grad   = ATmap0(Rp1*sqrt(bscale*cscale));
               etaorg = norm(x*bscale- proxSortedL1(x*bscale - grad,lambda1org));               
               eta    = etaorg / (1 + norm(grad) + norm(x*bscale));
               gs     = sort(abs(grad),'descend');
               infeas = max(max(cumsum(gs-lambda1org)),0)/lambda1org(1);
               primobj = objscale*(0.5*norm(xi)^2 + lambda1'* sort(abs(x),'descend')); 
               dualobj = objscale*(-0.5*norm(xi)^2 -b'*xi + lambda1'*sort(abs(up),'descend')- up'*u);
               relgap = abs(primobj-dualobj)/max(1,abs(primobj));
               if  (eta < stoptol && infeas <stoptol && relgap <stoptol)
                  breakyes = 1;
                  msg = 'Strongly converged';
               end
            end
         end
      end
      
   if etime(clock, tstart) > 0.5*3600
      breakyes = 777;
      msg      = 'half hour,time out!';
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
         primobj = objscale*(0.5*norm(xi)^2 + lambda1'* sort(abs(x),'descend'));
         dualobj = objscale*(-0.5*norm(xi)^2 - b'*xi);
         relgap  = abs(primobj-dualobj)/max( 1,abs(primobj));
         ttime   = etime(clock,tstart);
        if (printyes)
           fprintf('\n %5.0d| [%3.2e %3.2e] [%3.2e %3.2e]  %- 3.2e| %- 10.9e %- 10.9e |',...
               iter,primfeas,dualfeas,primfeasorg, dualfeasorg,relgap,primobj,dualobj); 
           fprintf(' %5.1f| %3.2e|',ttime, sigma);                
           fprintf('%2.3f|',gamma); 
           if strcmp(Asolver,'cg')
               fprintf('[%3.0d %3.0d]', length(resnrmxi)-1, solve_okxi);
           end
           if exist('eta') 
              fprintf('\n [eta = %3.2e, etaorg = %3.2e]',...
                  eta, etaorg); 
           end
        end
	    if (rem(iter,5*print_iter)==1) 
           normx    = norm(x); 
           normAtxi = norm(Atxi);  
           normu    = norm(u);
           if (printyes)
              fprintf('\n        [normx,Atxi,u =%3.2e %3.2e %3.2e]',...
              normx,normAtxi, normu);
           end
        end
        runhist.primobj(iter) = primobj;
        runhist.dualobj(iter) = dualobj;
        runhist.time(iter)    = ttime; 
        runhist.relgap(iter)  = relgap;
     end    
     if (breakyes > 0) 
        fprintf('\n  breakyes = %3.1f, %s',breakyes,msg); 
        break; 
     end    
     
 
 %% update sigma
 
      if (maxfeas < 5*stoptol) %% important 
         use_infeasorg = 1;
      end   
      if (use_infeasorg)
         feasratio                  = primfeasorg/dualfeasorg;  
         runhist.feasratioorg(iter) = feasratio; 
      else
         feasratio                  = primfeas/dualfeas;           
         runhist.feasratio(iter)    = feasratio; 
      end      
      if (feasratio < 1)
         prim_win = prim_win+1; 
      else
         dual_win = dual_win+1; 
      end   
      sigma_update_iter = sigma_fun(iter); 
      sigmascale        = 1.25; 
      sigmaold          = sigma;
      if (~sig_fix) && (rem(iter,sigma_update_iter)==0) 
   	     sigmamax = 1e6; sigmamin = 1e-8; 
	     if (iter <= 1*2500) 
	        if (prim_win > max(1,1.2*dual_win)) 
               prim_win = 0; 
               sigma    = min(sigmamax,sigma*sigmascale);
	        elseif (dual_win > max(1,1.2*prim_win)) 
               dual_win = 0; 
               sigma    = max(sigmamin,sigma/sigmascale); 
            end
         else
            if (use_infeasorg)
               feasratiosub  = runhist.feasratioorg([max(1,iter-19):iter]);
            else 
               feasratiosub  = runhist.feasratio([max(1,iter-19):iter]);                
            end
            meanfeasratiosub = mean(feasratiosub);
            if meanfeasratiosub < 0.1 || meanfeasratiosub > 1/0.1
               sigmascale    = 1.4; 
            elseif meanfeasratiosub < 0.2 || meanfeasratiosub > 1/0.2
               sigmascale    = 1.35;
            elseif meanfeasratiosub < 0.3 || meanfeasratiosub > 1/0.3
               sigmascale    = 1.32;
            elseif meanfeasratiosub < 0.4 || meanfeasratiosub > 1/0.4
               sigmascale    = 1.28;
            elseif meanfeasratiosub < 0.5 || meanfeasratiosub > 1/0.5
               sigmascale    = 1.26;
            end
            primidx   = find(feasratiosub <= 1); 
            dualidx   = find(feasratiosub >  1); 
            if (length(primidx) >= 12) 
               sigma  = min(sigmamax,sigma*sigmascale);
            end
            if (length(dualidx) >= 12)
               sigma  = max(sigmamin,sigma/sigmascale); 
            end
         end
      end
      if abs(sigmaold - sigma) > eps
         if strcmp(Asolver,'cg')
            parxi.sigma = sigma;
            AAtxi       = (IpsigAAtxi - xi)/sigmaold;
            IpsigAAtxi  = xi + sigma*AAtxi;
         elseif exist('Lxi','var')
            Lxi         = mychol(eye(m) + sigma*AAt,m);
         end
      end
   end

%% recover orignal variables
   if (iter == maxiter)
      msg           = ' maximum iteration reached';
      info.termcode = 3;
   end
   xi   = xi*sqrt(bscale*cscale);
   Atxi = ATmap0(xi);
   x    = x*bscale;
   u    = u*cscale;
   if ~exist('up','var'); up = x; end
   up   = up*bscale;
   Ax   = Ax*sqrt(bscale*cscale);
   Rd   = Atxi + u;
   Rp1  = Ax - borg;
   
   normRp      = norm(Rp1 - xi);
   normRd      = norm(Rd);
   normu       = norm(u); 
   primfeasorg = normRp/normborg;
   dualfeasorg = normRd/(1 + normu);
   primobj     =  0.5*norm(xi)^2 + lambda1org'* sort(abs(x),'descend');
   dualobj     = -(0.5*norm(xi)^2 + borg'*xi); 
   primobjorg  = 0.5*norm(Ax - borg)^2 +lambda1org'* sort(abs(x),'descend'); 
   relgap      = abs(primobj-dualobj)/max( 1,abs(primobj)); 
   obj         = [primobj, dualobj];
   if iter > 0
      grad     = ATmap0(Ax - borg);  
      etaorg   = norm(x*bscale- proxSortedL1(x*bscale - grad,lambda1org));       
      eta      = etaorg/(1+norm(grad)+norm(x));
   else
      etaorg   = nan;
      eta      = nan;
   end
   runhist.m   = m;
   runhist.n   = n;
   ttime       = etime(clock,tstart);
   ttime_cpu   = cputime - tstart_cpu;
   ttCG        = sum(runhist.psqmrxiiter);
   
   runhist.iter       = iter;
   runhist.totaltime  = ttime;
   runhist.primobjorg = primobj; 
   runhist.dualobjorg = dualobj;
   runhist.maxfeas    = max([dualfeasorg, primfeasorg]);
   runhist.etaorg     = etaorg;
   runhist.eta        = eta;
   info.m             = m;
   info.n             = n;
   info.minx          = min(min(x));
   info.maxx          = max(max(x));
   info.relgap        = relgap;
   info.ttCG          = ttCG;
   info.iter          = iter;
   info.time          = ttime;
   info.time_cpu      = ttime_cpu;
   info.sigma         = sigma;
   info.etaorg        = etaorg;
   info.eta           = eta;
   info.bscale        = bscale;
   info.cscale        = cscale;
   info.objscale      = objscale;
   info.dualfeasorg   = dualfeasorg;
   info.primfeasorg   = primfeasorg;
   info.obj           = obj;
   info.nnzx          = cardcal(x,0.999);
   if phase2 == 1
     info.Ax   = Ax;
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
      fprintf('\n     primobj = %10.9e, dualobj = %10.9e, relgap = %3.2e',primobj,dualobj, relgap);  
      fprintf('\n  primobjorg = %10.9e',primobjorg);
      fprintf('\n  primfeasorg  = %3.2e, dualfeasorg = %3.2e',...
	      primfeasorg, dualfeasorg); 
      if iter >= 1; fprintf('\n  Total CG number = %3.0d, CG per iter = %3.1f', ttCG, ttCG/iter);end
      fprintf('\n  eta = %3.2e, etaorg = %3.2e', eta, etaorg);
      fprintf('\n  min(X)    = %3.2e, max(X)    = %3.2e',...
          info.minx,info.maxx); 
      fprintf('\n  number of nonzeros in x (0.999) = %3.0d', cardcal(x,0.999));
      fprintf('\n--------------------------------------------------------------');
      fprintf('------------------\n');
   end
%%**********************************************************************
% update sigma
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
      elseif (iter < inf)  
         sigma_update_iter = 100;
      end
%%**********************************************************************

  

