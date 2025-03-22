function [eta_list,obj_list] = compute_sol_quality(X,lambda_list,funf,gradf,funp,R_fun)
nlambda = length(lambda_list);
eta_list = zeros(nlambda,1);
obj_list = zeros(nlambda,1);
for i = 1:nlambda
    lambda1 = lambda_list(i);
    x = X(:,i);
    grad = gradf(x);
    R_value = R_fun(x,lambda1);
    eta_list(i) = norm(R_value)/(1+norm(grad)+norm(x));
    obj_list(i) = funf(x) + funp(x,lambda1);
end
end