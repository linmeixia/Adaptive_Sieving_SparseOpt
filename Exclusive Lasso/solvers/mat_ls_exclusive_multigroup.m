function y = mat_ls_exclusive_multigroup(xi,par,nzcol)
AP1 = par.AP1;
D1 = par.D1;
const_vec = par.const_vec;
counts = par.counts;

tmp = (xi'*AP1)';
tmp2 = par.smat*(D1.*tmp);
tmp2 = repelem(const_vec.*tmp2,counts);
tmp2 = tmp-tmp2.*D1;
y = xi + par.sigma*(AP1*tmp2);
end

% function y = mat_ls_exclusive_multigroup(xi,par,nzcol)
% P = par.group_info.P;
% PT = par.group_info.PT;
% M = par.group_info.M;
% Amap = par.Amap;
% ATmap = par.ATmap;
% rr1 = par.info_u.rr1; 
% D = par.info_u.D;
% n = par.n;
% rho = par.rho;
% 
% tmp = ATmap(xi);
% tmp = tmp(P);
% [~,mm] = size(M);
% 
% x = zeros(n,1);
% for i = 1:mm
%     xtmp = tmp(M(1,i):M(2,i));
%     rtmp = rr1(M(1,i):M(2,i)); 
%     cr = ~rtmp;
%     dcr = cr.*D(M(1,i):M(2,i));
%     const = 2*rho*(dcr'*xtmp)/(1+sum(cr)*2*rho);
%     xtmp = cr.*xtmp - const*dcr;
%     x(M(1,i):M(2,i)) = xtmp;
% end
% tmp = x(PT);
% y = xi + par.sigma*Amap(tmp);
% end
% 
