function y = mat_ppa_logistic_exclusive_onegroup(xi,par,nzcol)
AP1 = par.AP1;
AP2 = par.AP2;
const = par.const;

tmp1 = (xi'*AP1)';
tmp2 = xi'*AP2;

tmptmp = par.info_w.r;
y = par.sigdtau*xi.*tmptmp + par.sigma*(AP1*tmp1-(const*tmp2)*AP2);
end