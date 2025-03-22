function [lb,ub,dd] = subgradient_grouplasso(x,c,ind)
lb = sign(x);
ub = sign(x);
idx = find(x==0);
lb(idx) = -1;
ub(idx) = 1;
lb = lb*c(1);
ub = ub*c(1);

dd = zeros(size(ind,2),1);
for i = 1:size(ind,2)
   xtmp = x(ind(1,i):ind(2,i));
   if norm(xtmp) == 0
       dd(i) = c(2)*ind(3,i);
   else
       lb(ind(1,i):ind(2,i)) = lb(ind(1,i):ind(2,i)) + (c(2)*ind(3,i)/norm(xtmp))*xtmp;
       ub(ind(1,i):ind(2,i)) = ub(ind(1,i):ind(2,i)) + (c(2)*ind(3,i)/norm(xtmp))*xtmp;
   end
end
end