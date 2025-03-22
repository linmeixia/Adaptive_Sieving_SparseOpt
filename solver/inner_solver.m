function all_var = inner_solver(Ainput,b,n,lossname,regularizer,para,op,all_var,solvername)
if strcmp(lossname,'least squares')
    if strcmp(regularizer.name,'lasso')
        solver_list = {'SSNAL','ADMM','APG','GRNM'};
        if ~any(strcmp(solver_list,solvername)); error('Pleaes import this solver for solving the regularized problem.');  end
        if strcmp(solvername,'SSNAL')
            if ~isfield(all_var, 'initial')
                [~,u,xi,x,~] = Classic_Lasso_SSNAL(Ainput,b,n,para,op);
                all_var.initial = 1;
            else
                [~,u,xi,x,~] = Classic_Lasso_SSNAL(Ainput,b,n,para,op,all_var.u,all_var.xi,all_var.x);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi; 
        elseif strcmp(solvername,'ADMM')
            if ~isfield(all_var, 'initial')
                [~,u,xi,x,~] = admmL1S(Ainput,b,n,para,op);
                all_var.initial = 1;
            else
                [~,u,xi,x,~] = admmL1S(Ainput,b,n,para,op,all_var.u,all_var.xi,all_var.x);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi;
        elseif strcmp(solvername,'APG')
            if ~isfield(all_var, 'initial')
                [~,x,~,~] = apg_ls(Ainput,b,n,regularizer,para,op);
                all_var.initial = 1;
            else
                [~,x,~,~] = apg_ls(Ainput,b,n,regularizer,para,op,all_var.x);
            end
            all_var.x = x; 
        elseif strcmp(solvername,'GRNM')
            if isstruct(Ainput)
                error('GRNM is not implemented for the linear map A without the matrix representations');
            end
            if ~isfield(all_var, 'initial')
                [x,~] = GRNM_ls_lasso(Ainput,b,n,para,op);
                all_var.initial = 1;
            else
                [x,~] = GRNM_ls_lasso(Ainput,b,n,para,op,all_var.x);
            end
            all_var.x = x; 
        end
    elseif strcmp(regularizer.name,'exclusive lasso')
        solver_list = {'SSNAL','ADMM','APG'};
        if ~any(strcmp(solver_list,solvername)); error('Pleaes import this solver for solving the regularized problem.'); end
        if strcmp(solvername,'SSNAL')
            if ~isfield(all_var, 'initial')
                [~,x,xi,u,~] = nal_exclusivelasso(Ainput,b,n,para,op.group_info,op);
                all_var.initial = 1;
            else
                [~,x,xi,u,~] = nal_exclusivelasso(Ainput,b,n,para,op.group_info,op,all_var.x,all_var.xi,all_var.u);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi;
        elseif strcmp(solvername,'ADMM')
            if ~isfield(all_var, 'initial')
                [~,xi,u,x,~] = admm_Exclusivelasso(Ainput,b,n,para,op.group_info,op);
                all_var.initial = 1;
            else
                [~,xi,u,x,~] = admm_Exclusivelasso(Ainput,b,n,para,op.group_info,op,all_var.x,all_var.xi,all_var.u);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi;
        elseif strcmp(solvername,'APG')
            if ~isfield(all_var, 'initial')
                [~,x,~,~] = apg_ls(Ainput,b,n,regularizer,para,op);
                all_var.initial = 1;
            else
                [~,x,~,~] = apg_ls(Ainput,b,n,regularizer,para,op,all_var.x);
            end
            all_var.x = x; 
        end
    elseif strcmp(regularizer.name,'sparse group lasso')
        solver_list = {'SSNAL','ADMM','APG'};
        if ~any(strcmp(solver_list,solvername)); error('Pleaes import this solver for solving the regularized problem.'); end
        if strcmp(solvername,'SSNAL')
            if ~isfield(all_var, 'initial')
                [~,xi,u,x,~] = SGLasso_SSNAL(Ainput,b,n,para*regularizer.corg,op.G,op.ind,op);
                all_var.initial = 1;
            else
                [~,xi,u,x,~] = SGLasso_SSNAL(Ainput,b,n,para*regularizer.corg,op.G,op.ind,op,all_var.xi,all_var.u,all_var.x);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi;
        elseif strcmp(solvername,'ADMM')
            if isstruct(Ainput)
                error('ADMM for sparse group lasso is not implemented for the linear map A without the matrix representations');
            end
            P = Def_P(n,op.G,op.ind);
            if ~isfield(all_var, 'initial')
                [xi,u,x,~]= GL_DADMM_nol(Ainput,b,para*regularizer.corg,P,op);
                all_var.initial = 1;
            else
                [xi,u,x,~]= GL_DADMM_nol(Ainput,b,para*regularizer.corg,P,op,all_var.xi,all_var.u,all_var.x);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi;
        elseif strcmp(solvername,'APG')
            if ~isfield(all_var, 'initial')
                [~,x,~,~] = apg_ls(Ainput,b,n,regularizer,para,op);
                all_var.initial = 1;
            else
                [~,x,~,~] = apg_ls(Ainput,b,n,regularizer,para,op,all_var.x);
            end
            all_var.x = x; 
        end
    elseif strcmp(regularizer.name,'slope')
        solver_list = {'SSNAL','ADMM','APG'};
        if ~any(strcmp(solver_list,solvername)); error('Pleaes import this solver for solving the regularized problem.'); end
        if strcmp(solvername,'SSNAL')
            if ~isfield(all_var, 'initial')
                [~,x,xi,u,~] = Newt_ALM(Ainput,b,n,para*op.lambda_BH',op);
                all_var.initial = 1;
            else
                [~,x,xi,u,~] = Newt_ALM(Ainput,b,n,para*op.lambda_BH',op,all_var.x,all_var.xi,all_var.u);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi;
        elseif strcmp(solvername,'ADMM')
            if ~isfield(all_var, 'initial')
                [~,xi,u,x,~] = ADMM(Ainput,b,n,para*op.lambda_BH',op);
                all_var.initial = 1;
            else
                [~,xi,u,x,~] = ADMM(Ainput,b,n,para*op.lambda_BH',op,all_var.x,all_var.xi,all_var.u);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi;
        elseif strcmp(solvername,'APG')
            if ~isfield(all_var, 'initial')
                [~,x,~,~] = apg_ls(Ainput,b,n,regularizer,para,op);
                all_var.initial = 1;
            else
                [~,x,~,~] = apg_ls(Ainput,b,n,regularizer,para,op,all_var.x);
            end
            all_var.x = x; 
        end
    end
elseif strcmp(lossname,'logistic')
    if strcmp(regularizer.name,'lasso')
        solver_list = {'SSNAL'};
        if ~any(strcmp(solver_list,solvername)); error('Pleaes import this solver for solving the regularized problem.'); end
        if strcmp(solvername,'SSNAL')
            if ~isfield(all_var, 'initial')
                [~,x,xi,u,~,~] = ppa_logistic_lasso(Ainput,b,n,para,op);
                all_var.initial = 1;
            else
                [~,x,xi,u,~,~] = ppa_logistic_lasso(Ainput,b,n,para,op,all_var.x,all_var.xi,all_var.u);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi;
        end
    elseif strcmp(regularizer.name,'exclusive lasso')
        solver_list = {'SSNAL'};
        if ~any(strcmp(solver_list,solvername)); error('Pleaes import this solver for solving the regularized problem.'); end
        if strcmp(solvername,'SSNAL')
            if ~isfield(all_var, 'initial')
                [~,x,xi,u,~,~] = ppa_logistic_exclusive(Ainput,b,n,para,op.group_info,op);
                all_var.initial = 1;
            else
                [~,x,xi,u,~,~] = ppa_logistic_exclusive(Ainput,b,n,para,op.group_info,op,all_var.x,all_var.xi,all_var.u);
            end
            all_var.x = x; all_var.u = u; all_var.xi = xi;
        end
    else
        error('Solver for %s and %s is not available.', lossname, regularizer.name);
    end
end
% nnz(all_var.x)
end