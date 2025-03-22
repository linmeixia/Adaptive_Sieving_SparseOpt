function y = mat_ppa_lasso_logistic(xi,par,nzcol)
Amap = par.Amap;
ATmap = par.ATmap;
rr1 = par.info_u; 

tmp = ATmap(xi);
tmp = rr1.*tmp;
y = par.sigma*Amap(tmp);

tmptmp = par.info_w.r;
y = par.sigdtau*xi.*tmptmp + y;
end