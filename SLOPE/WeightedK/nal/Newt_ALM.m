%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Use Newt_ALM to solve the SLOPE model:
%% min { 0.5*||Ax - b||^2 + kappa_lambda (x) 
%% from the dual pespective: 
%% - min {0.5*||xi||^2 + b'*xi + kappa^*_lambda (u): A'*xi + u = 0}
%% where kappa_lambda(x)=lambda'*sort(abs(x),'descend')  
%% and kappa^*_lambda is the conjugate function, [m,n]=size(A) with m<n
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [obj,x,xi,u,info,runhist] = Newt_ALM(Ainput,b,n,lambda,options,x0,xi0,u0)
% Algorithm 1: Newt_ALM
% Initialization: (1) use inputs x0,xi0,u0 if any; 
%                 (2) by default: x0,xi0,u0 are all zeros;
%                 (3) the numerical solution obtianed by ADMM with low
%                     accuracy in Phase I (not a good choice)
% Augmented parameter sigma tuning: ranging from [sigmamin, sigmamax] and
%                                   scaled by factors sigmascale (=2) or
%                                   1/sigmascale_reduce (1/sqrt(2))
%                                   based on the changes of primal and
%                                   dual infeasibilities
% Stopping criteria: KKT residual; infeasibility; relative duality gap
%                    The accuracy parameter stoptol (1e-6 by default)

   rng('default');
   maxiter       = 50000;
   stoptol       = 1e-6;
   printyes      = 1;
   printminoryes = 1;
   startfun      = @ADMM;
   Sd            = @(x) sort(abs(x),'descend');  
   
   sigma0        = 1;
   admm.iter     = 0;
   admm.time     = 0;
   admm.timecpu  = 0;
   use_proximal  = 1;
   scale         = 1;
   rescale       = 1; 
   precond       = 2;
   stopop        = 2;
   gapcon        = 1;
   printsub      = 1;
   runphaseI     = 0; 
   
   if isfield(options,'maxiter');       maxiter       = options.maxiter;            end
   if isfield(options,'stoptol');       stoptol       = options.stoptol;            end
   if isfield(options,'printyes');      printyes      = options.printyes;           end
   if isfield(options,'printminoryes'); printminoryes = options.printminoryes;      end
   if isfield(options,'sigma');         sigma0        = options.sigma;              end
   if isfield(options,'rescale');       rescale       = options.rescale;            end
   if isfield(options,'use_infeasorg'); use_infeasorg = options.use_infeasorg;      end
   if isfield(options,'use_proximal');  use_proximal  = options.use_proximal;       end
   if isfield(options,'Lip');           Lip           = options.Lip;                end
   if isfield(options,'precond');       precond       = options.precond;            end
   if isfield(options,'printsub');      printsub      = options.printsub;           end
   if isfield(options,'runphaseI');     runphaseI     = options.runphaseI;          end
%% Amap and ATmap
   tstart     = clock;
   tstart_cpu = cputime;
   m          = length(b);
   if isstruct(Ainput)
      if isfield(Ainput,'A');     A      = Ainput.A;     end
      if isfield(Ainput,'Amap');  Amap0  = Ainput.Amap;  end
      if isfield(Ainput,'ATmap'); ATmap0 = Ainput.ATmap; end
   else
      A      = Ainput; 
      Amap0  = @(x) A*x;
      ATmap0 = @(y) A'*y;
   end
   AATmap0   = @(x) Amap0(ATmap0(x));
   if ~exist('Lip','var')
      eigsopt.issym = 1;
      tstartLip     = clock;
      Lip           = eigs(AATmap0,m,1,'LA',eigsopt);
      fprintf('\n Lip = %3.2e, time = %3.2f', Lip, etime(clock, tstartLip));
   end
   sigmaLip   = 1/sqrt(Lip);
   lambdaorg = lambda;
   borg       = b;
   normborg   = 1 + norm(borg);
   if ~exist('x0','var') || ~exist('xi0','var') || ~exist('u0','var') 
      x = zeros(n,1); xi = zeros(m,1); u = zeros(n,1);
   else
      x = x0; xi = xi0; u = u0;
   end
   
%% phase I
   admm.op.stoptol = 1e-2;  
   admm.op.maxiter = 100;
   admm.op.sigma   = min(1,sigmaLip);
   admm.op.phase2  = 1;
   admm.op.use_infeasorg = 0;
   if admm.op.maxiter > 0 && runphaseI
      [obj,xi,u,x,info_admm,runhist_admm] = startfun(Ainput,b,n,lambdaorg,admm.op,x,xi,u);
      admm.xi0     = xi; 
      admm.u0      = u; 
      admm.x0      = x; 
      admm.Atxi0   = info_admm.Atxi; 
      admm.Ax0     = info_admm.Ax;
      Atxi         = admm.Atxi0; 
      Ax           = admm.Ax0; 
      admm.iter    = admm.iter + info_admm.iter;
      admm.time    = admm.time + info_admm.time;
      admm.timecpu = admm.timecpu + info_admm.time_cpu;
      bscale       = info_admm.bscale;
      cscale       = info_admm.cscale;
      objscale     = info_admm.objscale;
      if info_admm.eta < stoptol
         fprintf('\n Problem solved in Phase I \n');
         fprintf('\n ADMM Iteration No. = %3.0d, ADMM time = %3.1f s \n', admm.iter, admm.time);
         info              = info_admm;
         info.m            = m;
         info.n            = n;
         info.minx         = min(min(x));
         info.maxx         = max(max(x));
         info.relgap       = info_admm.relgap;
         info.iter         = 0;
         info.time         = admm.time;
         info.time_cpu     = admm.timecpu;
         info.admmtime     = admm.time;
         info.admmtime_cpu = admm.timecpu;
         info.admmiter     = admm.iter;
         info.eta          = info_admm.eta;
         info.etaorg       = info_admm.etaorg;
         info.obj          = obj;
         info.maxfeas      = max([info_admm.dualfeasorg, info_admm.primfeasorg]);
         runhist           = runhist_admm;
         return
      end
   else
      Atxi     = ATmap0(xi); 
      Ax       = Amap0(x); 
      obj(1)   = 0.5*norm(Ax - borg)^2 + lambdaorg'*Sd(x); 
      obj(2)   = -(0.5*norm(xi)^2 + borg'*xi);
      bscale   = 1;
      cscale   = 1;
      objscale = bscale*cscale;
   end
   %%

   if scale == 1
      b     = b/sqrt(bscale*cscale);
      xi    = xi/sqrt(bscale*cscale);
      Amap  = @(x) Amap0(x*sqrt(bscale/cscale));
      ATmap = @(x) ATmap0(x*sqrt(bscale/cscale));
      if exist('A','var'); A = A*sqrt(bscale/cscale);  end
      lambda = lambda/cscale;  
      x       = x/bscale; 
      u       = u/cscale; 
      Ax      = Ax/sqrt(bscale*cscale);
      Atxi    = Atxi/cscale; 
      normb   = 1 + norm(b);
   end
   Ainput_nal.Amap  = Amap;
   Ainput_nal.ATmap = ATmap;
   if exist('A','var'); Ainput_nal.A = A;              end
   sigma       = min(1, sigmaLip);
   if isfield(options,'sigma'); sigma = options.sigma; end
   Rp1         = Ax - b;
   Rd          = Atxi + u;
   normu       = norm(u);
   normRp      = norm(Rp1 - xi);
   normRd      = norm(Rd);
   primfeas    = normRp/normb;
   dualfeas    = normRd/(1+normu);
   maxfeas     = max(primfeas,dualfeas);
   dualfeasorg = normRd*cscale/(1+normu*cscale);
   primfeasorg = sqrt(bscale*cscale)*normRp/normborg;
   maxfeasorg  = max(primfeasorg, dualfeasorg);
   relgap      = (obj(1) - obj(2))/max(1,abs(obj(1)));
   runhist.dualfeasorg(1) = dualfeasorg;
   
   if printyes
        fprintf('\n *******************************************************');
        fprintf('******************************************');
        fprintf('\n \t\t   Phase II: Newt_ALM ');
        fprintf('\n ******************************************************');
        fprintf('*******************************************\n');
        if printminoryes
            fprintf('\n bscale = %3.2e, cscale = %3.2e', bscale, cscale);
            fprintf('\n ---------------------------------------------------');
        end
        fprintf('\n  iter|  [pinfeas  dinfeas]  [pinforg  dinforg]    relgaporg|    pobj          dobj    |');
        fprintf(' time | sigma |');
        fprintf('\n*****************************************************');
        fprintf('**************************************************************');
        fprintf('\n #%3.1d|  %3.2e %3.2e %3.2e %3.2e %- 3.2e %- 8.7e %- 8.7e  %5.1f',...
           0,primfeas,dualfeas,primfeasorg,dualfeasorg,relgap,obj(1),obj(2),etime(clock,tstart)); 
        fprintf('  %3.2e ',sigma);
   end
   
%% semi-smooth Newton conjugate gradient (ssncg) method
   SSNCG = 1;
   if SSNCG
      parNCG.matvecfname = 'mvSLOPE';
      parNCG.sigma       = sigma;
      parNCG.tolconst    = 0.5;
      parNCG.n           = n;
      parNCG.precond     = precond;
   end
   gamma       = 1;
   maxitersub  = 10;
   breakyes    = 0;
   prim_win    = 0;
   dual_win    = 0;
   RpGradratio = 1;
   ssncgop.tol = stoptol;
   
   ssncgop.precond  = precond;
   ssncgop.bscale   = bscale;
   ssncgop.cscale   = cscale;
   ssncgop.printsub = printsub;
   for iter = 1:maxiter           
      if ((rescale == 1) && (maxfeas < 5e2) && (rem(iter,3) == 1) && (iter > 1) )...
         || ((rescale >= 2) && maxfeas < 1e-1 && (abs(relgap) < 0.05) ...
             && (iter >= 5) && (max(normx/normuxi,normuxi/normx) > 1.7) && rem(iter,5)==1)
         normAtxi = norm(Atxi);
         normx    = norm(x); 
         normu    = norm(u);
         normuxi  = max([normAtxi,normu]); 
         if normx < 1e-7; normx = 1; normuxi = 1; end
         const    = 1;
         bscale2  = normx*const;
         cscale2  = normuxi*const;
         sbc      = sqrt(bscale2*cscale2);
         b        = b/sbc; 
         lambda  = lambda/cscale2;
         x        = x/bscale2;
         
         Ainput_nal.Amap  = @(x) Ainput_nal.Amap(x*sqrt(bscale2/cscale2));
         Ainput_nal.ATmap = @(x) Ainput_nal.ATmap(x*sqrt(bscale2/cscale2));
         if isfield(Ainput_nal,'A'); Ainput_nal.A = Ainput_nal.A*sqrt(bscale2/cscale2); end
         if precond == 2 && isfield(parNCG,'dA'); parNCG.dA = parNCG.dA*bscale2/cscale2; end
         xi               = xi/sbc; 
         Atxi             = Atxi/cscale2; Ax = Ax/sbc;
         u                = u/cscale2; 
         sigma            = sigma*(cscale2/bscale2);
         cscale           = cscale*cscale2;
         bscale           = bscale*bscale2;
         objscale         = objscale*(cscale2*bscale2);
         ssncgop.bscale   = bscale;
         ssncgop.cscale   = cscale;
         normb            = 1 + norm(b);
         if printyes
            fprintf('\n    ');
            fprintf('[rescale=%1.0f: %2.0f| %3.2e %3.2e %3.2e | %3.2e %3.2e| %3.2e]',...
            rescale,iter,normx,normAtxi,normu,bscale,cscale,sigma);
         end         
         rescale  = rescale+1; 
         prim_win = 0; 
         dual_win = 0; 
      end 
      xold = x;  
      uold = u;
      parNCG.sigma     = sigma;
      parNCG.innerNT   = 0;
      parNCG.innerflsa = 0;
      if dualfeas < 1e-5
         maxitersub = max(maxitersub,30);
      elseif dualfeas < 1e-3
         maxitersub = max(maxitersub,30);
      elseif dualfeas < 1e-1
         maxitersub = max(maxitersub,20);
      end
      ssncgop.maxitersub = maxitersub; 
      [u,Atxi,xi,parNCG,runhist_NCG,info_NCG] = NCGWK(b,Ainput_nal,x,Ax,Atxi,xi,...
                  lambda,parNCG,ssncgop);
      if info_NCG.breakyes < 0
         parNCG.tolconst = max(parNCG.tolconst/1.06,1e-3);
      end
      x   = info_NCG.up;
      Ax  = info_NCG.Aup;
      Rd  = Atxi + u;
      Rp1 = Ax - b;
     
      normRp      = norm(Rp1 - xi);
      normRd      = norm(Rd);
      normu       = norm(u);
      primfeas    = normRp/normb;
      dualfeas    = normRd/(1+normu);
      maxfeas     = max(primfeas,dualfeas);
      dualfeasorg = normRd*cscale/(1+normu*cscale);
      primfeasorg = sqrt(bscale*cscale)*normRp/normborg;
      maxfeasorg  = max(primfeasorg, dualfeasorg);
      
      runhist.dualfeas(iter+1)    = dualfeas;
      runhist.primfeas(iter+1)    = primfeas;
      runhist.maxfeas(iter+1)     = maxfeas;
      runhist.primfeasorg(iter+1) = primfeasorg;
      runhist.dualfeasorg(iter+1) = dualfeasorg;
      runhist.maxfeasorg(iter+1)  = maxfeasorg;
      runhist.sigma(iter)         = sigma;
      runhist.rank2(iter)         = sum(parNCG.info_u.rr2);
      runhist.innerNT(iter)       = parNCG.innerNT;
      runhist.innerflsa(iter)     = parNCG.innerflsa;
      runhist.xr(iter)            = sum(abs(x)>1e-10);

     
    
      primobj = objscale*(0.5*norm(xi)^2 +lambda'*Sd(x));   
      dualobj = objscale*(-0.5*norm(xi)^2 - b'*xi );
      relgap  = abs(primobj-dualobj)/max(1,abs(primobj)); 
      ttime   = etime(clock,tstart);

 %% check for termination
         if stopop == 1
            if (max([primfeasorg,dualfeasorg]) < 500*max(1e-6, stoptol)) 
                grad   = ATmap0(Rp1*sqrt(bscale*cscale));
                etaorg = norm(x*bscale-proxSortedL1(x*bscale-grad,lambdaorg)); 
                eta    = etaorg / (1 + norm(grad) + norm(x*bscale));
                if eta < stoptol 
                   breakyes = 1;
                   msg = 'KKT residual converged';
                elseif abs(relgap) < stoptol && max([primfeasorg,dualfeasorg]) < stoptol && eta < sqrt(stoptol)
                   breakyes = 2;
                   msg = 'Relative gap & KKT residual converged';
                end 
            end
         elseif stopop == 2 
            if max([primfeasorg,dualfeasorg]) < stoptol
               grad   = ATmap0(Rp1*sqrt(bscale*cscale)); 
               etaorg = norm(x*bscale-proxSortedL1(x*bscale-grad,lambdaorg)); 
               eta    = etaorg / (1 + norm(grad) + norm(x*bscale));
               gs     = sort(abs(grad),'descend');
               infeas = max(max(cumsum(gs-lambdaorg)),0)/lambdaorg(1);
               if (eta< stoptol && abs(relgap) < stoptol) %&& infeas< stoptol)
                  breakyes = 999;
                  msg = 'Relative gap & KKT residual converged';
               end  
            end
         end
         if (printyes)
            fprintf('\n %5.0d| [%3.2e %3.2e] [%3.2e %3.2e]  %- 3.2e| %- 10.9e %- 10.9e |',...
               iter,primfeas,dualfeas,primfeasorg, dualfeasorg,relgap,primobj,dualobj); 
            fprintf(' %5.1f| %3.2e|',ttime, sigma); 
            if iter >= 1
               fprintf('%3d|',sum(parNCG.info_u.rr2));
              fprintf('[%3d ]',sum(abs(x) > 1e-10));  
            end
            fprintf(' sigmaorg = %3.2e, ', sigma*(bscale/cscale));
            if exist('eta'); fprintf('\n \t [ eta = %3.2e, etaorg = %3.2e]',eta, etaorg);end
         end
         if (rem(iter,3*1)==1)
            normx    = norm(x); 
            normAtxi = norm(Atxi); 
            normu    = norm(u);
            if (printyes)
               fprintf('\n        [normx,Atxi,u =%3.2e %3.2e %3.2e ]',...
               normx,normAtxi,normu);
            end
         end
         runhist.primobj(iter)   = primobj;
         runhist.dualobj(iter)   = dualobj;
         runhist.time(iter)      = ttime; 
         runhist.relgap(iter)    = relgap;
      
     if (breakyes > 0) 
        if printyes; fprintf('\n  %s',msg);  end
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
      
      sigmascale        = 2;  %5^0.5;%2;
      sigmascale_reduce = 2^0.5;
      sigmamax          = 1e8;
      if (rem(iter,sigma_update_iter)==0) || info_NCG.breakyes >= 0
   	     sigmamin = 1e-4; 
	     if prim_win > max(1,1.2*dual_win) && (info_NCG.breakyes < 0)
            prim_win = 0;
            sigma = min(sigmamax,sigma*sigmascale);
         elseif (dual_win > max(1,3*prim_win) ||  info_NCG.breakyes >= 0)
            dual_win = 0;
            sigma = max(sigmamin, sigma/sigmascale_reduce);
         end
      end
   end
 
%% recover orignal variables
   if (iter == maxiter)
      msg   = ' maximum iteration reached';
      info.termcode = 3;
   end
   ttime    = etime(clock,tstart);
   
   xi       = xi*sqrt(bscale*cscale);
   Atxi     = ATmap0(xi);
   u        = u*cscale;
   x        = x*bscale;
   Ax       = Ax*sqrt(bscale*cscale);
   Rd       = Atxi + u;
   Rp1      = Ax - borg;
   normRp   = norm(Rp1 - xi);
   normRp1  = norm(Rp1);
   normRd   = norm(Rd);
   normu    = norm(u);
   
   primfeasorg = normRp/normborg;
   dualfeasorg = normRd/(1+normu);
   primobj     = 0.5*norm(xi)^2 + lambdaorg'*Sd(x); 
   dualobj     = -(0.5*norm(xi)^2 + borg'*xi);
   primobjorg  = 0.5*normRp1^2 + lambdaorg'*Sd(x); 
   relgap      = abs(primobj-dualobj)/max(1,abs(primobj));
   obj         = [primobj, dualobj];
   grad        = ATmap0(Rp1);
   etaorg      = norm(x-proxSortedL1(x-grad,lambdaorg));
   eta         = etaorg/(1+norm(grad)+norm(x));
   gs          = sort(abs(grad),'descend');
   infeas      = max(max(cumsum(gs-lambdaorg)),0)/lambdaorg(1);
  
   runhist.m          = m;   
   runhist.n          = n;
   ttime_cpu          = cputime - tstart_cpu;
   runhist.iter       = iter;
   runhist.totaltime  = ttime;
   runhist.primobjorg = primobj; 
   runhist.dualobjorg = dualobj;
   runhist.maxfeas    = max([dualfeasorg, primfeasorg]);
   runhist.eta        = eta;
   runhist.etaorg     = etaorg;
   runhist.infeas     = infeas;
   info.infeas        = infeas;    
   info.m             = m;
   info.n             = n;
   info.minx          = min(min(x));
   info.maxx          = max(max(x));
   info.relgap        = relgap;
   info.iter          = iter;
   info.time          = ttime;
   info.time_cpu      = ttime_cpu;
   info.admmtime      = admm.time;
   info.admmtime_cpu  = admm.timecpu;
   info.admmiter      = admm.iter;
   info.eta           = eta;
   info.etaorg        = etaorg;
   info.obj           = obj;
   info.dualfeasorg   = dualfeasorg;
   info.primfeasorg   = primfeasorg;
   info.maxfeas       = max([dualfeasorg, primfeasorg]);
   info.Axmb          = normRp1;
   info.nnzx          = cardcal(x,0.999);
   info.x             = x;
   info.u             = u;
   info.xi            = xi;
   if (printminoryes) 
       fprintf('\n--------------------------------------------------------------');
       fprintf('----------------------------');
      if runphaseI
          fprintf('\n  Phase I:   ADMM Iteration No. = %3.0d,  ADMM time = %3.1f s,  ADMM time per iter = %5.4f s', admm.iter, admm.time,admm.time/admm.iter);
          fprintf('\n  Phase II:  ALM Iteration No.  = %2.0d,  ALM time = %3.2f s,   ALM time per iter  = %5.4f s',iter,ttime,ttime/iter); 
      else
          fprintf('\n  ALM Iteration No. = %2.0d,  ALM time = %3.2f s,  ALM time per iter = %5.4f s',iter,ttime,ttime/iter);
      end
      fprintf('\n  cputime = %3.2f s', ttime_cpu);
      fprintf('\n  primobj =  %10.9e,   dualobj = %10.9e, relgap = %3.2e, d_infeas = %9.2e',primobj,dualobj,relgap,infeas);   
      fprintf('\n  primfeasorg = %3.2e, dualfeasorg  = %3.2e',...
	      primfeasorg, dualfeasorg); 
      fprintf('\n  eta    = %3.2e,  etaorg = %3.2e', eta, etaorg);
      fprintf('\n  min(X) = %3.2e, max(X) = %3.2e',...
          info.minx,info.maxx); 
      fprintf('\n  number of nonzeros in x (0.999) = %3.0d', cardcal(x,0.999));     
      fprintf('\n------------------------------------------------------------------------------------------\n'); 
   end

   