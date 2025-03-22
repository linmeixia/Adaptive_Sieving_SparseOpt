function [regularizer,op] = compute_info_of_regularizer(A,n,group_info,regularizer)
op = [];
if strcmp(regularizer.name,'exclusive lasso')
    org_group = zeros(n,1);
    for j = 1:size(group_info.M,2)
        org_group(group_info.P(group_info.M(1,j):group_info.M(2,j))) = j;
    end
    group_info.org_group = org_group;
    regularizer.group_info = group_info;
    op.group_info = group_info;
elseif strcmp(regularizer.name,'sparse group lasso')
    A = A(:,group_info.P);
    regularizer.sorted_group = group_info.org_group(group_info.P);
    regularizer.G = 1:size(A,2);
    regularizer.ind = zeros(3,group_info.group_num);
    for j = 1:group_info.group_num
        regularizer.ind(1,j) = group_info.M(1,j);
        regularizer.ind(2,j) = group_info.M(2,j);
        regularizer.ind(3,j) = sqrt(group_info.M(2,j)-group_info.M(1,j)+1);
    end
    regularizer.corg = [1,1/max(regularizer.ind(3,:))];
    regularizer.P = Def_P(n,regularizer.G,regularizer.ind);
    op.G = regularizer.G;
    op.ind = regularizer.ind;
    op.P = regularizer.P;
elseif strcmp(regularizer.name,'slope')
    regularizer.lambda_BH = norminv(1-(1:n)*(0.1/(2*n)));
    op.lambda_BH = regularizer.lambda_BH;
end
end