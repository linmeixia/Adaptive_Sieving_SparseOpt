Authors: Yancheng Yuan, Meixia Lin, Defeng Sun, and Kim-Chuan Toh. 

A MATLAB software designed to use adaptive sieving (a dimension reduction technique) for sparse optimization problems

         min  Phi(x) + P(x)
      where Phi: R^n -> R is a convex twice continuously differentiable function, 
            P: R^n -> (-infty,infty] is a closed and proper convex function.

In particular, we consider the following setting:
(1) Phi(x) = h(Ax), where A is a given m by n data matrix, h: R^m -> R is a twice continuously differentiable loss function taking one of:
    (1.1) (linear regression) h(y) = sum_{i=1}^m (y_i - b_i)^2/2, for some given vector b in R^m;
    (1.2) (logistic regression) h(y) = sum_{i=1}^m log(1+exp(-b_i y_i)), for some given vector b in {-1,1}^m.
(2) P(.) is one of:
    (2.1) (lasso) P(x) = lambda*||x||_1, where lambda>0 is a given parameter;
    (2.2) (sparse group lasso) P(x) = lambda1*||x||_1 + lambda2*sum_{l=1}^g w_l*||x_{G_l}||, where lambda1,lambda2>0, w_1,...,w_g>=0 are parameters, {G_1,...,G_g} is a disjoint partition of {1,...,n};
    (2.3) (exclusive lasso) P(x) = lambda*sum_{l=1}^g ||w_{G_l} .* x_{G_l}||_1^2, where lambda>0, w in R^n is a positive weight vector, {G_1,...,G_g} is a disjoint partition of {1,...,n};
    (2.4) (Sorted L-One Penalized Estimation or SLOPE) P(x) = sum_{i=1}^n lambda_i |x|_{(i)}, where lambda_1>=lambda_2>=...>=lambda_n>=0 and lambda_1>0, and |x|_{(i)} means the ith largest component of |x|.

=========================================================================================================================
The software contains six folds:

    solver: main function: Adaptive_sieving.m and other utility functions

    Data: sample dataset and data generation functions
    
    Lasso:  downloaded from SuiteLasso (http://www.lixudong.info/software/suitelasso/)

    Sparse group lasso:  downloaded from SparseGroupLasso (https://github.com/YangjingZhang/SparseGroupLasso)
                          
    SLOPE: downloaded from SLOPE-Solver--Newt_ALM (https://github.com/LuoZiyanBJTU/SLOPE-Solver--Newt_ALM)

    Exclusive Lasso: downloaded from exclusive-lasso-solver (https://github.com/linmeixia/exclusive-lasso-solver)

=========================================================================================================================
Guideline:

(1) set up in Matlab:
>> Startup

(2) run test file:
>> run_test

=========================================================================================================================

Citation:
If you use this work, please cite:

Yancheng Yuan, Meixia Lin, Defeng Sun, and Kim-Chuan Toh, “Adaptive sieving: A dimension reduction technique for sparse optimization problems”, Mathematical Programming Computation, 2025, to appear.



