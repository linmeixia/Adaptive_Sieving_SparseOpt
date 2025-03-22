function z = findzero(a,b,y)
z = 0;
maxiter = 100;
ab = a*b;
for i = 1:maxiter
    expbz = exp(b*z);
    h = z-y-ab/(1+expbz);
    if abs(h)<1e-12
        break;
    end
    hdiff = 1+a*expbz/(1+expbz)^2;
    z = z-h/hdiff; 
    if isnan(z)
        break;
    end
end
end


% function z = findzero(a,b,y)
% z = 0;
% maxiter = 100;
% ab = a*b;
% 
% if (y<0) && (b<0) && ((y+ab/2)<-30)
%     z = y;
%     return;
% end
% expbz = exp(b*z);
% h = z-y-ab/(1+expbz);
% for i = 1:maxiter
%     alp = 1;
%     maxit = 10;
%     zold = z;
%     hold = h;
%     if isnan(expbz) || (expbz>1e12)
%         hdiff = 1;
%     else
%         hdiff = 1+a*expbz/(1+expbz)^2;
%     end
%     dz = -h/hdiff;    
%     for ii = 1:maxit
%         z = zold + alp*dz;
%         expbz = exp(b*z);
%         if isnan(expbz) || (expbz>1e12)
%             h = z-y;
%         else
%             h = z-y-ab/(1+expbz);
%         end
%         if abs(h)<abs(hold)
%             break;
%         else
%             alp = alp/2;
%         end
%     end
%     if abs(h)<1e-12
%         return;
%     end
% end
% end

% function z = findzero(a,b,y)
% z = 0;
% maxiter = 100;
% ab = a*b;
% if (y<0) && (b<0) && ((y+ab/2)<-30)
%     z = y;
%     return;
% end
% for i = 1:maxiter
%     expbz = exp(b*z);
%     h = z-y-ab/(1+expbz);
%     hdiff = 1+a*expbz/(1+expbz)^2;
%     z = z-h/hdiff; 
% end
% end