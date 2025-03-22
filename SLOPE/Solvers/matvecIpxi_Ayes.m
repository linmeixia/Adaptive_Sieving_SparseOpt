function Ay = matvecIpxi_Ayes(y,par,AP)
tmp = AP'*y;
Ay = y + par.sigma*(AP*tmp);