function [x,info] = proxSortedL1(y,lambda)
% The function is modified based on the function from SLOPE Toolbox version 1.0.
% Copyright 2013, M. Bogdan, E. van den Berg, W. Su, and E.J. Candes
%    The SLOPE Toolbox is free software: you can redistribute it
%    and/or  modify it under the terms of the GNU General Public License
%    as published by the Free Software Foundation, either version 3 of
%    the License, or (at your option) any later version.
%
%    The SLOPE Toolbox is distributed in the hope that it will
%    be useful, but WITHOUT ANY WARRANTY; without even the implied
%    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%    See the GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with the SLOPE Toolbox. If not, see
%    <http://www.gnu.org/licenses/>.

% Normalization
lambda  = lambda(:);
y       = y(:); 
sgn     = mexssgn(y); % position at zero element will be valued 1
[y,idx] = sort(abs(y),'descend');

% Simplify the problem
k       = find(y > lambda,1,'last'); % locate the last element in y larger than lambda

% Compute solution and re-normalize
n       = numel(y);
x       = zeros(n,1);

rr2     = ones(n,1);
Bmap    = @(x) x - [x(2:end);0];
nv      = [];

if (~isempty(k))
   v1   = y(1:k);
   v2   = lambda(1:k);
   v    = proxSortedL1Mex(v1,v2);
   nv   = (v>1e-20);
   
   x(idx(1:k)) = v;
   vv          = Bmap(v); 
   rr2([1:k])  = (abs(vv)<1e-12);
end

% Restore signs
x        = sgn .* x;
info.idx = idx;
info.rr2 = rr2;
info.k   = k;
info.s   = sgn;
info.nz  = sum(nv);


 