%%****************************************************************
%% solve the linear system (I + sigma*A*J*At)xi = rhs
%%****************************************************************

function [xi,resnrm,solve_ok,par] = WKsolve(Ainput,rhs,par)
options  = 2; 
resnrm   = 0; 
solve_ok = 1;
rr2      = par.info_u.rr2;   
m        = length(rhs); 
if false   
   [h,U] = WKJacobian(rr2);
else
   [h,U] = mexWK(double(rr2));
end

op   = 2;
oopp = 0;

 if oopp == 1
     Idnn = eye(length(par.info_u.rr2));
     if ~isempty(U) 
         VV2 = Ainput.A*diag(par.info_u.s)*Idnn(par.info_u.idx,:)'*U;  
     else
         VV2 = [];
     end
     VV1     = Ainput.A*diag(par.info_u.s)*Idnn(par.info_u.idx,:)'*diag(h);
 end 
      
  
 
vidx  = find(h>0);
v1idx = par.info_u.idx(vidx);
V1    = Ainput.A(:,v1idx);  %  A*pi'*diag(h)
  
  
if ~isempty(U)
     uu    = sum(U,2);
     idx_U = find(uu>0); % nonzero row index set
     Unew  = U(idx_U,:); % dropping zeros rows from U
     iidex = find(par.info_u.s<0);
     idx_A = par.info_u.idx(idx_U);
     if op == 1
        sgnidx = intersect(iidex,idx_A);
        tmp1   = -Ainput.A(:,sgnidx);
        Ainput.A(:,sgnidx) = tmp1; 
        Ahat   = Ainput.A(:,idx_A);
        V2     = Ahat*Unew; % A*pi'*U
     else
       Ahat        = Ainput.A(:,idx_A);
       [~,~,idA]   = intersect(iidex,idx_A);
       Ahat(:,idA) = -Ahat(:,idA);
       V2          = Ahat*Unew;
     end
  else
     V2 = [];
  end
 
  
par.lenP    = size(V1,2);  
par.numblk1 = size(V2,2);  
 
  
  
if (par.lenP + par.numblk1 > 6e3 && m > 6e3) || (  par.lenP + par.numblk1 >1e4 && m > 1000)
     options = 1;
end
if options == 1
    par.V1 = V1;
    par.V2 = V2;
    par.precond = 0;
    [xi,~,resnrm,solve_ok] = psqmry('mvWK',Ainput,rhs,par);
  else
     if m<(par.lenP + par.numblk1)
        if ~isempty(V2)
            tmpM1 = par.sigma*(V1*V1'+V2*V2');
        else
            tmpM1 = par.sigma*(V1*V1');
        end
        M         = speye(m,m) + tmpM1;
        L         = mychol(M,length(M));      
        xi        = mylinsysolve(L,rhs);  
     else
  %% woodbury formula
        W         = [V1 V2];
        nW        = size(W,2) ;
        SMWmat    = W'*W;      
        SMWmat    = spdiags(ones(nW,1)/par.sigma,0,nW,nW)+ SMWmat; 
        L         = mychol(SMWmat,nW);
        xi        = rhs - W*mylinsysolve(L,(rhs'*W)');  
     end
  end
   par.innerop = options;

    
    
    
    
 
 
 
 
 
 
   
 
 
 
 
 
 
 

