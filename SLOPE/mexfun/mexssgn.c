//
//  mexssgn.c
//  oscar
//
//  Created by Lin Meixia on 2017/5/11.
//  Copyright © 2017年 Lin Meixia. All rights reserved.
//

/***********************************************************************
 * mexBfun.c : C mex file
 *
 *   z = mexssgn(x);
 *
 *   z(i) = 1 if x(i)>=0
 *   z(i) = -1 if x(i)<0
 ***********************************************************************/

#include <mex.h>
#include <math.h>
#include <matrix.h>

#ifndef MWSIZE_MAX
#define  mwIndex        int
#define  mwSignedIndex  int
#define  mwSize         int
#endif

/**********************************************************
 *
 ***********************************************************/
void mexFunction(int nlhs, mxArray  *plhs[],
                 int nrhs, const mxArray  *prhs[] )

{    double   *x, *z;
    double   *xtmp;
    mwIndex  *irx, *jcx;
    mwSize   nx, ny, j, k, r, l;
    double   tmp, mu;
    
    /* CHECK FOR PROPER NUMBER OF ARGUMENTS */
    
    if (nrhs <1){
        mexErrMsgTxt("ssgn: requires at least 1 input arguments."); }
    if (nlhs > 1){
        mexErrMsgTxt("ssgn: requires 1 output argument."); }
    
    /* CHECK THE DIMENSIONS */
    
    nx = mxGetM(prhs[0]);
    if (mxIsSparse(prhs[0])) {
        irx = mxGetIr(prhs[0]);  jcx = mxGetJc(prhs[0]); xtmp = mxGetPr(prhs[0]);
        x = mxCalloc(nx,sizeof(double));
        for (k=0; k<jcx[1]; ++k) { r=irx[k]; x[r]=xtmp[k]; }  }
    else {
        x = mxGetPr(prhs[0]);
    }
    /***** Do the computations in a subroutine *****/
    plhs[0] = mxCreateDoubleMatrix(nx,1,mxREAL);
    z = mxGetPr(plhs[0]);
    
    l = 0;
    for (j=0; j<nx; ++j) {
        if (x[j]>=0){
            z[j]=1;
        } else {
            z[j]=-1;
        }
    }
    return;
}
/**********************************************************/
