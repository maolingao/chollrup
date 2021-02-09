
#include <math.h>
#include "mex.h"
#include "mex_helper.h"
#include "blas_headers.h"

char errMsg[200];

/* Main function CHOLUPRK1S */

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    
    /* Read arguments */
    if (nrhs<4)
      mexErrMsgTxt("Not enough input arguments");
    if (nlhs>1)
      mexErrMsgTxt("Too many return arguments");
    if (mxIsCell(prhs[0])) 
        mexErrMsgTxt("Too many return arguments");
    
    // definition
    int i,sz,ione=1,retcode=0;
//     const double* vvec;
    double *vvec, *cvec, *svec;
    double *tbuff;
    double temp;
    mwIndex *ir, *jc;
    mwSize nzmax;
    int m, n;
    mxDouble *pr;
    int nnz;
    
    // input
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    pr = mxGetPr(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    nzmax = mxGetNzmax(prhs[0]);
    nnz = jc[n];
//     printf("[choluprk1s]: input (%d,%d) with nzmax=%d; nnz=%d; pr[0]=%f; pr[end]=%f.\n",m,n,nzmax,nnz,pr[0],pr[nnz-1]);
    
    // vvec, cvec, svec should have same dim as row/column of factor
    if (getVecLen(prhs[1],"VEC")<n) mexErrMsgTxt("VEC too short");
    vvec=mxGetPr(prhs[1]);
    if (getVecLen(prhs[2],"CVEC")!=n || getVecLen(prhs[3],"SVEC")!=n)
        mexErrMsgTxt("CVEC, SVEC have wrong size");
    cvec=mxGetPr(prhs[2]); svec=mxGetPr(prhs[3]);
    
    // factor must be square matrix
    if (m!=n){
        sprintf(errMsg,"[choluprk1s] Matrix has wrong size (%d,%d).", m, n);
        mexErrMsgTxt(errMsg);
    }
    
    // factor must have positive diagonal
    for (i=0,tbuff=pr; i<n; i++) {

        if (*tbuff<=0.0) {
            sprintf(errMsg,"[choluprk1s] lfact is not positive definite *tbuff=%f.",*tbuff);
            mexErrMsgTxt(errMsg);
            retcode=1; break;
        }
        // printf("[choluprk1s] *tbuff=%f. jc[%d]=%d.\n",*tbuff,i,jc[i]);
        tbuff = pr + (jc[i+1]);
    }
    
//     // working vector, such that vvec is not modified.
//     BLASFUNC(dcopy) (&n,vvec,&ione,wkvec,&ione);
    // array of fst_vector to store columns of the updated factor
    fst_vector tbuff_sparse[n];
    
    // Givens rotations
    int numNonZeros;
    mwIndex* ircol;
    for (i=0,sz=n,tbuff=pr; i<n-1; i++) {
    // printf("[choluprk1s] processing column %d. \n", i+1);

    // for each column
    // convert the sparse column to full (progressive shrinking)
    double tbuff_full[sz];
    memset(tbuff_full, 0, sz*sizeof(double));   // initialise tbuff_full to all zero
    numNonZeros = jc[i+1]-jc[i];                // numNonZeros of current column i
    ircol = &ir[jc[i]];                         // pointer to the start of the ir of current column i
    sparse2full(tbuff,i,numNonZeros,ircol,tbuff_full);
//     printf("[choluprk1s] compute Givens using (%f,%f)\n \t numNonZeros=%d, tbuff[0]=%f.\n",tbuff_full[0],*(vvec+i),numNonZeros,(tbuff[0]));
    
    // compute Givens
    BLASFUNC(drotg) (tbuff_full,vvec+i,cvec+i,svec+i);
    
    /* Do not want negative elements on factor diagonal */
    if ((temp=*tbuff_full)<0.0) {
        *tbuff_full=-temp; cvec[i]=-cvec[i]; svec[i]=-svec[i];
    } else if (temp==0.0) {
        sprintf(errMsg,"Zero eigenvalue of new factor.");
        mexErrMsgTxt(errMsg);
        retcode=1; break;
    }

    // apply Givens to the rest of dense columns
    sz--;
    BLASFUNC(drot) (&sz,tbuff_full+1,&ione,vvec+(i+1),&ione,cvec+i,svec+i);

    // convert the full column to sparse
    full2sparse(tbuff_full, n, sz+1, &(tbuff_sparse[i]));

    // logistics
    tbuff+=numNonZeros;
    }

    // the very last Givens
    tbuff_sparse[i].buff = (double*) mxMalloc(1*sizeof(double));
    tbuff_sparse[i].buff[0] = tbuff[0];
    tbuff_sparse[i].ir = (mwIndex*) mxMalloc(1*sizeof(mwIndex));
    tbuff_sparse[i].ir[0]=n-1; // index starting from 0. thus -1
    tbuff_sparse[i].nnz=1;
    if (tbuff_sparse[i].buff == NULL || tbuff_sparse[i].ir == NULL)                     
    {
        printf("Error! memory not allocated for the last fst_vector.");
        exit(0);
    }
//     printf("[choluprk1s] last Givens (%f,%f)\n",tbuff_sparse[i].buff[0],vvec[n-1]);
    
    if (retcode==0 && (*tbuff!=0.0 || vvec[i]!=0.0)) {
        BLASFUNC(drotg) (tbuff_sparse[i].buff,vvec+i,cvec+i,svec+i);
//         printf("[choluprk1s] done last Givens, L[end]=%f, vvec[end]=%f, cvec[end]=%f, svec[end]=%f.\n", *(tbuff_sparse[i].buff), *(vvec+i), *(cvec+i), *(svec+i) );
        if ((temp=*tbuff)<0.0) {
          *tbuff=-temp; cvec[i]=-cvec[i]; svec[i]=-svec[i];
        } else if (temp==0.0) retcode=1;
    } else retcode=1;
    
    // construct the updated factor
    fst_matrix lmat_update;
    fst_matrix_init(tbuff_sparse, n, &lmat_update);
//     printf("[choluprk1s]: created new factor (%d,%d), nnz=%d.\n",lmat_update.m,lmat_update.n,lmat_update.nnz);

    /*  show all entries in the sparse matrix  */
//     for(int ii=0; ii<lmat_update.nnz; ii++){
//       int ncol=0;
//       for (int iii=0; iii<lmat_update.n+1; iii++){
//           if (lmat_update.jc[iii]<=ii){
//               ncol = iii+1;
//           }
//       }
//       printf("[choluprk1s]: lfact(%d,%d)=%f.\n",lmat_update.ir[ii]+1,ncol,lmat_update.buff[ii]);
//     }
      
    // modify the input
    mxSetNzmax(prhs[0],lmat_update.nnz);
    mxSetPr(prhs[0], mxRealloc(pr, lmat_update.nnz*sizeof(double)));
    mxSetIr(prhs[0], mxRealloc(ir, lmat_update.nnz*sizeof(mwIndex)));

    pr  = mxGetPr(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);

    memcpy(pr, lmat_update.buff, lmat_update.nnz*sizeof(double));
    memcpy(ir, lmat_update.ir, lmat_update.nnz*sizeof(mwIndex));
    memcpy(jc, lmat_update.jc, (n+1)*sizeof(mwIndex));
    
    // return msg
    if (nlhs==1) {
    plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
    *(mxGetPr(plhs[0]))=(double) retcode;
    }
      
}