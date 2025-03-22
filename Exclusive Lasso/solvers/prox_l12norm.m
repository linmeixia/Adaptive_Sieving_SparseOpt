%% Analytical Solution for the following optimization problem with vector input w
%% argmin_x 1/2||x - w||^2 + \frac{rho}{2}\|x\|_1^2
%% analytical solution

function [x, sign_info, act_idx] = prox_l12norm(w, rho)
n = size(w,1);
sign_info = sign(w);
% sign_info(sign_info>=0)=1;
w = abs(w);
w2 = sort(w, 'descend');
wsum = cumsum(w2);
alptmp = wsum./(1/rho+[1:n]');
alp = max(alptmp);
x = max(w-alp,0);
act_idx = (x==0);
x = x.*sign_info;
end






