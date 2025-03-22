%%**************************************************************
%% Jacobian = spdiags(hh,0,n,n) + U*U';
%%**************************************************************
   function [hh,U] = WKJacobian(rr2)
        
    n      = length(rr2); % SLOPE case
    blklen = []; 
    blkend = [];
    len    = 0; 
    numblk = 0;            
       for k=1:length(rr2)
          if (rr2(k)==1)  
             len = len+1; 
          else
             if (len > 0) 
                numblk           = numblk+1;
                blklen(numblk,1) = len;
                blkend(numblk,1) = k; 
                len              = 0;                    
             end
          end
       end
       if (len > 0)
          numblk           = numblk+1;
          blklen(numblk,1) = len;
          blkend(numblk,1) = n; 
       end
   
%%    
    numblk   = length(blklen);
    if numblk == 0
        hh   = ones(n,1);
        U    = [];
    else
        U    = [];
        hh   = ones(n,1);      
    for k  = 1:numblk
        if (blkend(k)<n || len ==0) % k\neq N or N\notin J
           Len         = blklen(k)+1; 
           invsqrtlen  = 1/sqrt(Len);  
           idxend      = blkend(k); 
           idxsub      = [idxend-blklen(k): idxend];
           hh(idxsub)  = 0; 
           vv          = zeros(n,1);
           vv(idxsub)  = invsqrtlen; %%ones(len,1)/sqrt(len); 
           U           = [U vv];                                         
        else
            idxsub     = [n-blklen(numblk)+1: n]; % K=N and N\in J
            hh(idxsub) = 0; 
        end
    end
    end
    

    
        
        
