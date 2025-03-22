function [z,info] = proximal_mapping_logistic(y,b,alpha)
m = length(b);
z = zeros(m,1);
info.r = zeros(m,1);
for i = 1:m
    zold = z(i);
    z(i) = findzero(alpha,b(i),y(i));
    if isnan(z(i))
        fun = @(x) x-y(i)-alpha*b(i)/(1+exp(b(i)*x));
        z0 = zold;
        z(i) = fzero(fun,z0);
    end 
end
% tmp = exp(b.*z);
% tmp2 = tmp./(1+tmp).^2;
% info.r = 1./(1+alpha*tmp2);
zmy = z-y;
alpha2 = alpha^2;
tmp = alpha2*(b.*zmy) - alpha*(zmy).^2 + alpha2;
info.r = alpha2./tmp;
end

