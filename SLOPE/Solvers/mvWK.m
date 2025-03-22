function WKxi = mvWK(xi,par,A)
V1 = par.V1;
V2 = par.V2;
WKxi = xi;

if ~isempty(V1)
   tmp1 = (xi'*V1)';
   WKxi = WKxi + par.sigma*V1*tmp1;
end

if ~isempty(V2)
   tmp2 = (xi'*V2)';
   WKxi = WKxi + par.sigma*V2*tmp2;
end

 