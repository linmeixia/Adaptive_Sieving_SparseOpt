/**********************************************************
* mexFusedLassoJacobian.c
*            
* [h,U] = mexFusedLassoJacobian(r); 
* Jacobian = spdiags(h,0,nr+1,nr+1)+U*U'; 
*
* mex  -O -largeArrayDims  mexWK.c
* Copyright (c) 2016 by
* Xudong Li, Defeng Sun, and Kim-Chuan Toh 
**********************************************************/

#include <mex.h>
#include <math.h>
#include <matrix.h>

#if !defined(MAX)
#define  MAX(A, B)   ((A) > (B) ? (A) : (B))
#endif
        
#if !defined(MIN)
#define  MIN(A, B)   ((A) < (B) ? (A) : (B))
#endif
        
#if !defined(SQR)
#define SQR(x) ((x)*(x))
#endif

/*#include "mymexheader.h"*/

/********************************************************************
  PROCEDURE mexFunction - Entry for Matlab
*********************************************************************/
void mexFunction(const int nlhs, mxArray *plhs[],
                 const int nrhs, const mxArray *prhs[])
{
  double   *rr, *vv, *hh;
  ptrdiff_t  *blklen, *blkend, *ii, *jj; 
  int      nr, len, numblk, NZ, idxstart, idxend, endflag;
  int      j, k, cnt; 
  double   tmp; 

  if(nrhs != 1)
    mexErrMsgTxt("mexWK: requires 1 input arguments.");
  if(nlhs != 2)
    mexErrMsgTxt("mexWK: requires 2 output argument.");
  if (mxIsLogical(prhs[0])) {
     mexErrMsgTxt("mexWK: input cannot be a logical array."); 
  }
  nr = MAX(mxGetM(prhs[0]),mxGetN(prhs[0])); 
  if (mxIsSparse(prhs[0])) {
     mexErrMsgTxt("mexWK: input cannot be sparse"); 
  } else {
     rr = mxGetPr(prhs[0]); 
  }
  if (nr == 0) {
     mexErrMsgTxt("mexWK: input size cannot be zero");
  }
  /*****************************/
   blklen = mxCalloc(nr,sizeof(mwSize));
   blkend = mxCalloc(nr,sizeof(mwSize));   
   len = 0; numblk = 0;  NZ = 0; endflag = 0;
   for (k=0; k<nr; k++) {
      if (rr[k]==1) { len++; 
      } else { 
        if (len > 0) {
           blklen[numblk] = len;
           blkend[numblk] = k;
           NZ = NZ + len + 1;
           numblk++;           
           len = 0; }         
      }
   }
   if (len > 0) {
      blklen[numblk] = len;
      blkend[numblk] = nr; 
      /*NZ += len;*/ 
      endflag = 1;
      numblk++;           
   }
  /* printf("\n numblk = %d",numblk);
   printf("\n blkend[0] = %d",blkend[0]);
   NZ += numblk; 
   printf("\n NZ = %d", NZ);*/
   plhs[0] = mxCreateDoubleMatrix(nr,1,mxREAL);
   if (endflag == 1){
      /*printf("\n endflag = %d", endflag);*/
      plhs[1] = mxCreateSparse(nr,numblk-1,NZ,mxREAL);
   } else {
      plhs[1] = mxCreateSparse(nr,numblk,NZ,mxREAL);
   }
 /************************************************
  ************************************************/
       
   hh = mxGetPr(plhs[0]);   
   ii = mxGetIr(plhs[1]);  
   jj = mxGetJc(plhs[1]);
   vv = mxGetPr(plhs[1]);  
   cnt = 0;
   if (numblk > 0) {
   for (k=0; k<numblk; k++) {
      idxend = blkend[k];
      if (idxend == nr){
         len = blklen[k];
         idxstart = idxend - len;
         for (j=idxstart; j<idxend; j++){
             hh[j] = 1;
         }
      }
      else {
           len = blklen[k];
           idxstart = idxend-len; 
           tmp = 1/sqrt(len+1);
           for (j=idxstart; j<=idxend; j++) { 
              hh[j] = 1; 
              ii[cnt] = j;      
              vv[cnt] = tmp;
              cnt++; 
           }
           jj[k+1] = cnt;
         }
   }
   }
   for (k=0; k<nr; k++) { 
      hh[k] = 1-hh[k]; 
   }
return;
}
/************************************************************************/

