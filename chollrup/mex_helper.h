/* -------------------------------------------------------------------
 * Helper functions for MEX functions
 * -------------------------------------------------------------------
 * Original Author: Matthias Seeger
 * Author: Maolin Gao
 * ------------------------------------------------------------------- */

#ifndef MEX_HELPER_H
#define MEX_HELPER_H

#include <math.h>
#include "mex.h"
#include <stdlib.h>
#include <string.h>

char errMsg[];

/*
 * The BLAS/LAPACK function xxx is called as xxx_ in Linux, but as
 * xxx in Windows. Uncomment the corresponding definition here.
 */
/* Version for Linux */
#define BLASFUNC(NAME) NAME ## _
/* Version for Windows */
/* #define BLASFUNC(NAME) NAME */

/*
 * Types for matrix arguments to support BLAS convention.
 * See 'parseBLASMatrix'.
 */
typedef struct {
  double* buff;
  int m,n;
  int stride;
  char strcode[4];
  mwSize nnz;          // sparse matrix only!
  mwSize nzmax;        // sparse matrix only!
  mwIndex *ir, *jc;    // sparse matrix only!
  bool isSparse;
} fst_matrix;

typedef struct {
  double* buff;
  int n;
  mwSize nnz;          // sparse matrix only!
  mwIndex *ir;         // sparse matrix only!
  bool isSparse;
} fst_vector;

void fst_matrix_init(fst_vector *tbuff_sparse, const int n, fst_matrix* lmat)
{
    mwSize nnz=0;
    for (int i=0; i<n; i++)
    {
        nnz += tbuff_sparse[i].nnz;
//         printf("[fst_matrix_init] nnz=%d.\n",(int)nnz);
    }
    
    // init and memory allocation
    lmat->nnz=nnz;
    lmat->isSparse=true;
    lmat->n=n;
    lmat->m=n;      // assuming square L!
    lmat->stride=n; // assuming square L!
    
    lmat->buff=(double*) mxMalloc(nnz * sizeof(double));
    lmat->ir=(mwIndex*) mxMalloc(nnz * sizeof(mwIndex));
    lmat->jc=(mwIndex*) mxMalloc((n+1) * sizeof(mwIndex));
    
    if (lmat->buff == NULL || lmat->ir == NULL || lmat->jc == NULL)                     
    {
        printf("Error! memory not allocated for fst_matrix.");
        exit(0);
    }
    mexMakeMemoryPersistent(lmat->buff);
    mexMakeMemoryPersistent(lmat->ir);
    mexMakeMemoryPersistent(lmat->jc);

    // content filling
    lmat->jc[0] = 0;
    for (int i=0; i<n; i++)
    {
//         printf("[fst_matrix_init] processing col=%d.\n",i+1);
        lmat->jc[i+1] = lmat->jc[i] + tbuff_sparse[i].nnz;
        memcpy((lmat->buff)+(lmat->jc[i]), tbuff_sparse[i].buff, tbuff_sparse[i].nnz*sizeof(double));
        memcpy((lmat->ir)+(lmat->jc[i]), tbuff_sparse[i].ir, tbuff_sparse[i].nnz*sizeof(mwIndex));
        
        // free memory in the temporary array of sparse columns
        mxFree(tbuff_sparse[i].buff);
        mxFree(tbuff_sparse[i].ir);
    }
}

void fst_vector_alloc(fst_vector *fvec, const int n, const mwSize nnz)
{
    fvec->n=n;
    fvec->nnz=nnz;
    fvec->isSparse=true;
    
    fvec->buff=(double*) mxMalloc(nnz * sizeof(double));
    fvec->ir=(mwIndex*) mxMalloc(nnz * sizeof(mwIndex));
    
    if (fvec->buff == NULL || fvec->ir == NULL)                      
    {
        printf("Error! memory not allocated for fst_vector.");
        exit(0);
    }
}

/*
 * Macros picking out specific positions from structure code array
 */
#define UPLO(arr) (arr)[0]

#define DIAG(arr) (arr)[2]

#define IsNonZero(d) (fabs(d)>=1e-6)
/*
 * Exported functions
 */

void sparse2full(const double* pr, const int offset, const int numNonZeros, const mwIndex* ir, double* pr_full)
{
    for (int i=0; i<numNonZeros; i++){
       pr_full[(*(ir+i)-offset)] = pr[i];
   }

}

void full2sparse(const double* lcol_full, const int n, const int n_to_rot, fst_vector* lcol)
{
    // memory estimation
    mwSize nnz=0;
    for (const double *ptr={lcol_full}; ptr<(lcol_full+n_to_rot); ptr++){
        if (IsNonZero(*ptr)){
//             printf("[full2sparse] *ptr=%f \n ", *ptr);
            nnz++;
        }
    }
    
    // memory allocation
    fst_vector_alloc(lcol, n, nnz);
    
    // content filling
    int i=0;
    int idx_row = 0;
    int idx_pivot = n-n_to_rot;
    for (const double *ptr={lcol_full}; ptr<(lcol_full+n_to_rot); ptr++){
        if (IsNonZero(*ptr)){
            lcol->buff[i] = *ptr;
            lcol->ir[i] = idx_pivot + idx_row;
//             printf("[full2sparse]: lcol(%d)=%f.\n",lcol->ir[i],lcol->buff[i]);
            i++;
        }
        idx_row++;
    }
    
}

double getScalar(const mxArray* arg,const char* name)
{
  if (!mxIsDouble(arg) || mxGetM(arg)!=1 || mxGetN(arg)!=1) {
    sprintf(errMsg,"Expect double scalar for %s",name);
    mexErrMsgTxt(errMsg);
  }
  return *mxGetPr(arg);
}

int getScalInt(const mxArray* arg,const char* name)
{
  double val,temp;

  if (!mxIsDouble(arg) || mxGetM(arg)!=1 || mxGetN(arg)!=1) {
    sprintf(errMsg,"Expect scalar for %s",name);
    mexErrMsgTxt(errMsg);
  }
  val=*mxGetPr(arg);
  if ((temp=floor(val))!=val) {
    sprintf(errMsg,"Expect integer for %s",name);
    mexErrMsgTxt(errMsg);
  }
  return (int) temp;
}

int getVecLen(const mxArray* arg,const char* name)
{
  int n;

  if (!mxIsDouble(arg)) {
    sprintf(errMsg,"Expect real vector for %s",name);
    mexErrMsgTxt(errMsg);
  }
  if (mxGetM(arg)==0 || mxGetN(arg)==0)
    return 0;
  if ((n=mxGetM(arg))==1)
    n=mxGetN(arg);
  else if (mxGetN(arg)!=1) {
    sprintf(errMsg,"Expect real vector for %s",name);
    mexErrMsgTxt(errMsg);
  }
  return n;
}

/*
 * NOTE: The string returned is allocated here using 'mxMalloc',
 * it has to be dealloc. by the user using 'mxFree'.
 */
const char* getString(const mxArray* arg,const char* name)
{
  int len;
  char* buff;

  if (!mxIsChar(arg) || mxGetM(arg)!=1) {
    sprintf(errMsg,"Expect char row vector for %s",name);
    mexErrMsgTxt(errMsg);
  }
  len=mxGetN(arg)+1;
  buff=(char*) mxMalloc(len*sizeof(char));
  mxGetString(arg,buff,len);

  return buff;
}

void checkMatrix(const mxArray* arg,const char* name,int m,int n)
{
  if (!mxIsDouble(arg)) {
    sprintf(errMsg,"Expect real matrix for %s",name);
    mexErrMsgTxt(errMsg);
  }
  if (m!=-1 && mxGetM(arg)!=m) {
    sprintf(errMsg,"Expect %d rows for %s",m,name);
    mexErrMsgTxt(errMsg);
  }
  if (n!=-1 && mxGetN(arg)!=n) {
    sprintf(errMsg,"Expect %d columns for %s",n,name);
    mexErrMsgTxt(errMsg);
  }
}

/*
 * Matrix argument can come with additional BLAS attributes. To pass these,
 * 'arg' must be a cell vector { BUFF, [YS XS M N], {SCODE} }. Here, BUFF
 * is a normal (buffer) matrix, YS, XS, M, N are integers, SCODE is a
 * string (optional). The matrix is BUFF(YS:(YS+M-1),XS:(XS+N-1)).
 *
 * Structure codes:
 * If SCODE is given, it must be a string of length 2. Pos.:
 * - 0: UPLO field, values 'U' (upper), 'L' (lower)
 * - 1: DIAG field, values 'U' (unit tri.), 'N' (non unit tri.)
 *      If UPLO spec., the def. value for DIAG is 'N'
 * A field value ' ' means: not specified.
 * These are passed to BLAS routines if required. The 'strcode' field
 * contains the codes separ. by 0, i.e. the C string for the codes
 * attached to each other.
 */
void parseBLASMatrix(const mxArray* arg,const char* name,
  fst_matrix* mat,int m,int n)
{
  int bm,bn,ys,xs,am,an,csz;
  const mxArray* bmat,*szvec,*scdvec;
  const double* iP;
  char sbuff[3];

  mat->strcode[0]=mat->strcode[2]=' ';
  mat->strcode[1]=mat->strcode[3]=0;
  if (!mxIsCell(arg)) {
    /* No cell array: Must be normal matrix */
//     printf("[parseBLASMatrix] Normal matrix\n");
    checkMatrix(arg,name,m,n);
    mat->buff=mxGetPr(arg);
    mat->stride=mat->m=mxGetM(arg); mat->n=mxGetN(arg);
    
    /* sparse matrix additionals */
    bmat = arg;
    if (mxIsSparse(bmat)){        
        mat->ir = mxGetIr(bmat);
        mat->jc = mxGetJc(bmat);
        mat->nzmax = mxGetNzmax(bmat);
        mat->nnz = mat->jc[mat->n];
        mat->isSparse = true;
//         printf("[parseBLASMatrix] sparse (%d,%d) matrix nnz=%d\n",mat->m,mat->n,mat->nnz);


    }
    else{
        mat->isSparse = false;
//         printf("[parseBLASMatrix] dense (%d,%d) matrix\n",mat->m,mat->n);
    }
    
    
  } else {
//     printf("[parseBLASMatrix] cell (%d,%d) matrix\n",mxGetM(arg),mxGetN(arg));
    if ((csz=mxGetM(arg)*mxGetN(arg))<2) {
      sprintf(errMsg,"Array %s has wrong size",name);
      mexErrMsgTxt(errMsg);
    }
    bmat=mxGetCell(arg,0);
    checkMatrix(bmat,name,-1,-1);
    bm=mxGetM(bmat); bn=mxGetN(bmat);
    if (getVecLen(mxGetCell(arg,1),name)!=4) {
      sprintf(errMsg,"Index vector in %s has wrong size",name);
      mexErrMsgTxt(errMsg);
    }
    iP=mxGetPr(mxGetCell(arg,1));
    ys=((int) iP[0])-1; xs=((int) iP[1])-1;
    am=(int) iP[2]; an=(int) iP[3];
    if (ys<0 || xs<0 || am<0 || an<0 || ys+am>bm || xs+an>bn) {
      sprintf(errMsg,"Index vector in %s wrong",name);
      mexErrMsgTxt(errMsg);
    }
    if ((m!=-1 && am!=m) || (n!=-1 && an!=n)) {
      sprintf(errMsg,"Matrix %s has wrong size",name);
      mexErrMsgTxt(errMsg);
    }
    /* fill mat: mat is a subblock of arg{0} */
    mat->buff=mxGetPr(bmat)+(xs*bm+ys);
    mat->m=am; mat->n=an; mat->stride=bm;
    mat->nzmax = mxGetNzmax(bmat);
    
    /* sparse matrix additionals */
    if (mxIsSparse(bmat)){        
        mat->ir = mxGetIr(bmat);
        mat->jc = mxGetJc(bmat);
        mat->nnz = mat->jc[mat->n];
        mat->isSparse = true;
//         printf("[parseBLASMatrix] sparse (%d,%d) matrix nnz=%d\n",mat->m,mat->n,mat->nnz);

    }
    else{
        mat->isSparse = false;
//         printf("[parseBLASMatrix] dense (%d,%d) matrix\n",mat->m,mat->n);
    }
    
    if (csz>2) {
      /* Structure codes */
      scdvec=mxGetCell(arg,2);
      if (!mxIsChar(scdvec) || mxGetM(scdvec)!=1 ||
	  mxGetN(scdvec)!=2) {
	sprintf(errMsg,"Structure code string in %s wrong",name);
	mexErrMsgTxt(errMsg);
      }
      mxGetString(scdvec,sbuff,3);
      if ((sbuff[0]!='U' && sbuff[0]!='L' && sbuff[0]!=' ') ||
	  (sbuff[1]!='U' && sbuff[1]!='N' && sbuff[1]!=' ')) {
	sprintf(errMsg,"Structure code string in %s wrong",name);
	mexErrMsgTxt(errMsg);
      }
      if (sbuff[0]!=' ' && sbuff[1]==' ')
	sbuff[1]='N'; /* def. value */
      if ((mat->m!=mat->n && sbuff[0]!=' ') ||
	  (sbuff[1]!=' ' && sbuff[0]==' ')) {
	sprintf(errMsg,"Structure code string in %s inconsistent",name);
	mexErrMsgTxt(errMsg);
      }
      mat->strcode[0]=sbuff[0]; mat->strcode[2]=sbuff[1];
    }
  }
  
}

void fillVec(double* vec,int n,double val)
{
  int i;
  for (i=0; i<n; i++) vec[i]=val;
}

bool isUndef(double val)
{
  return isnan(val) || (isinf(val)!=0);
}

#endif
