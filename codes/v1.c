#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "mmio.c"
#include <stdint.h>
#include <unistd.h>
//#include <sys/time.h>
#include <time.h>
#include <stdbool.h>

typedef struct block{
  int    * rowInd;  //[nz (of each block)]
  int    * colPtr;  //[b+1]
  int      nz;   //non zero elements of the block
  int      count;  //number of non zero elements of each column
  bool     zero; //true: block has only zeros, false: block has nz elements
} block;


void coo2csc(
  int       * const row,       /*!< CSC row start indices */
  int       * const col,       /*!< CSC column indices */
  int const * const row_coo,   /*!< COO row indices */
  int const * const col_coo,   /*!< COO column indices */
  int const         nnz,       /*!< Number of nonzero elements */
  int const         n,         /*!< Number of rows/columns */
  int const         isOneBased /*!< Whether COO is 0- or 1-based */
) {
  // ----- cannot assume that input is already 0!
  for (int l = 0; l < n+1; l++) col[l] = 0;

  // ----- find the correct column sizes
  for (int l = 0; l < nnz; l++)
    col[col_coo[l] - isOneBased]++;

  // ----- cumulative sum
  for (int i = 0, cumsum = 0; i < n; i++) {
    int temp = col[i];
    col[i] = cumsum;
    cumsum += temp;
  }
  col[n] = nnz;
  // ----- copy the row indices to the correct place
  for (int l = 0; l < nnz; l++) {
    int col_l;
    col_l = col_coo[l] - isOneBased;

    int dst = col[col_l];
    row[dst] = row_coo[l] - isOneBased;

    col[col_l]++;
  }
  // ----- revert the column pointers
  for (int i = 0, last = 0; i < n; i++) {
    int temp = col[i];
    col[i] = last;
    last = temp;
  }
}

void coo2csr(
  int       * const col,       /*!< CSR col start indices */
  int       * const row,       /*!< CSR row indices */
  int const * const row_coo,   /*!< COO row indices */
  int const * const col_coo,   /*!< COO column indices */
  int const         nnz,       /*!< Number of nonzero elements */
  int const         n,         /*!< Number of rows/columns */
  int const         isOneBased /*!< Whether COO is 0- or 1-based */
) {
  // ----- cannot assume that input is already 0!
  for (int l = 0; l < n+1; l++) row[l] = 0;

  // ----- find the correct row sizes
  for (int l = 0; l < nnz; l++)
    row[row_coo[l] - isOneBased]++;

  // ----- cumulative sum
  for (int i = 0, cumsum = 0; i < n; i++) {
    int temp = row[i];
    row[i] = cumsum;
    cumsum += temp;
  }
  row[n] = nnz;
  // ----- copy the col indices to the correct place
  for (int l = 0; l < nnz; l++) {
    int row_l;
    row_l = row_coo[l] - isOneBased;

    int dst = row[row_l];
    col[dst] = col_coo[l] - isOneBased;

    row[row_l]++;
  }
  // ----- revert the row pointers
  for (int i = 0, last = 0; i < n; i++) {
    int temp = row[i];
    row[i] = last;
    last = temp;
  }
}

void createF(int* rowIndF, int* colPtrF, int N){
  int* F = (int*)malloc(N*N*sizeof(int));
  int* rowFcoo = (int*)malloc(N*N*sizeof(int));
  int* colFcoo = (int*)malloc(N*N*sizeof(int));
  int nzF = 0;

  for(int i=0; i<N*N;i++){
    F[i] = 1;
  }

  for(int i=0; i<N;i++){
    for(int j=0; j<N; j++){
      if(F[N*i+j]==1){
        nzF++;
        colFcoo[nzF-1] = j;
        rowFcoo[nzF-1] = i;
      }
    }
  }

  coo2csc(rowIndF,colPtrF,rowFcoo,colFcoo,nzF,N,0);
}


void blocking(int* rowInd, int* colPtr, block* Blocks,int N,int nb,int b){
  int cols = 0;     //column of blocked array
  for(int i=b; i<(N+1);i=i+b){
    for(int j=colPtr[i-b]; j<colPtr[i]; j++){
        int index = rowInd[j]/b;      //row of blocked array

        Blocks[nb*cols+index].nz++;   //count nz of each block
        Blocks[nb*cols+index].rowInd = (int*) realloc(Blocks[nb*cols+index].rowInd,Blocks[nb*cols+index].nz*sizeof(int));
        Blocks[nb*cols+index].zero = false;  //includes nz elements
        Blocks[nb*cols+index].rowInd[Blocks[nb*cols+index].nz-1]=rowInd[j]%b;
    }
    cols++;
  }

  cols=0;

  for(int i=1; i<(N+1);i++){
    for(int j=colPtr[i-1]; j<colPtr[i]; j++){
      int index = rowInd[j]/b;
      Blocks[nb*cols+index].count++;
      for(int c=((i-1)%b)+1; c<(b+1); c++){
        Blocks[nb*cols+index].colPtr[c] += Blocks[nb*cols+index].count;
      }
      Blocks[nb*cols+index].count = 0;
    }

    if((i-1)%b==b-1 ){
      cols++;
    }
  }

  for(int i=0; i<nb*nb;i++){
    if(Blocks[i].zero){
      free(Blocks[i].rowInd);
      free(Blocks[i].colPtr);
    }
  }
}


void maskedMultiply(block F, bool* filter, block A, block B, block* C, int n, bool firstTime, int index){
  int size = 0;
  if(firstTime){
    for(int w=0; w<n; w++){                            //check for 1 on F_block
      for(int v=F.colPtr[w]; v<F.colPtr[w+1]; v++){
        int indexF = F.rowInd[v];
        for (int i=0; i<n; i++){                      //gia ton A
          for(int j=A.colPtr[i]; j<A.colPtr[i+1]; j++){
            int indexA = A.rowInd[j];
            if(indexA == indexF && filter[n*w+indexA] == true){
              for(int k=B.colPtr[w]; k<B.colPtr[w+1]; k++){
                int indexB = B.rowInd[k];
                if((indexB == i)){
                  filter[n*w+indexA] = false;
                  C->rowInd = (int*)realloc(C->rowInd, (size+1)*sizeof(int));
                  C->rowInd[size] = indexA;

                  size++;
                  for(int r=w+1; r<n+1;r++){
                    C->colPtr[r]++;
                  }
                  break;
                }
              }
              break;
            }
          }
        }
      }
    }
    C->nz = size;
  }else{
    size = C->nz;
    for(int w=0; w<n; w++){                          //check for 1 on F_block
      for(int v=F.colPtr[w]; v<F.colPtr[w+1]; v++){
        int indexF = F.rowInd[v];
        for (int i=0; i<n; i++){                      //gia ton A
          for(int j=A.colPtr[i]; j<A.colPtr[i+1]; j++){
            int indexA = A.rowInd[j];
            if(indexA == indexF && filter[n*w+indexA] == true){
              for(int k=B.colPtr[w]; k<B.colPtr[w+1]; k++){
                int indexB = B.rowInd[k];
                if((indexB == i)){
                  filter[n*w+indexA] = false;

                  C->rowInd = (int*)realloc(C->rowInd, (size+1)*sizeof(int));
                  C->rowInd[size] = indexA;
                  if(C->colPtr[w+1] - C->colPtr[w] == 0){
                    for(int t=size; t>C->colPtr[w+1]; t--){
                      int temp0 = C->rowInd[t];
                      C->rowInd[t] = C->rowInd[t-1];
                      C->rowInd[t-1] = temp0;
                    }
                  }else{
                    if(indexA > C->rowInd[C->colPtr[w+1]-1]){
                      for(int t=size; t>C->colPtr[w+1]+1; t--){
                        int temp1 = C->rowInd[t];
                        C->rowInd[t] = C->rowInd[t-1];
                        C->rowInd[t-1] = temp1;
                      }
                    }else{
                      for(int c=C->colPtr[w]; c<C->colPtr[w+1]; c++){
                        if(indexA < C->rowInd[c]){
                          for(int t=size; t>c; t--){
                            int temp2 = C->rowInd[t];
                            C->rowInd[t] = C->rowInd[t-1];
                            C->rowInd[t-1] = temp2;
                          }
                          break;
                        }
                      }
                    }
                  }

                  for(int r=w+1; r<n+1;r++){
                    C->colPtr[r]++;
                  }

                  size++;
                  break;
                }
              }
              break;
            }
          }
        }
      }
    }
    C->nz = size;
  }
}


void maskedMulBlockedArrays(block* F, block* A, block* B, block* C, int nb, int b){

  bool* filter = (bool*)malloc(b*b*sizeof(bool));
  for(int i=0; i<nb; i++){
    for(int j=0; j<nb; j++){
      bool firstTime = true;

      for(int f=0; f<b*b; f++){
        filter[f] = true;
      }

      for(int k=0; k<nb; k++){
        if((A[nb*k+i].zero == false && B[nb*j+k].zero == false) && (F[nb*j+i].zero == false)){
          maskedMultiply(F[nb*j+i],filter,A[nb*k+i],B[nb*j+k],&C[nb*j+i], b, firstTime,(nb*j+i));
          firstTime = false;
        }
      }
    }
  }
  free(filter);
}

int main(int argc, char *argv[]) {
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz, i, j, k, *coo_col, *coo_row;

    struct timespec ts_start,ts_end;

    if (argc < 2){
  	    fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
  	    exit(1);
    }
    else
        if ((f = fopen(argv[1], "r")) == NULL)
              exit(1);

    if (mm_read_banner(f, &matcode) != 0){
          printf("Could not process Matrix Market banner.\n");
          exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) ){
      printf("Sorry, this application does not support ");
      printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
      exit(1);
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      exit(1);


    /* reseve memory for matrices */

    coo_col = (int *) malloc(nz * sizeof(int));
    coo_row = (int *) malloc(nz * sizeof(int));

    if (!mm_is_pattern(matcode)){
        for (int i=0; i<nz; i++){
            fscanf(f, "%d %d\n", &coo_row[i], &coo_col[i]);
            coo_row[i]--;  /* adjust from 1-based to 0-based */
            coo_col[i]--;
        }
    }
    else {
        for (i=0; i<nz; i++){
            fscanf(f, "%d %d\n", &coo_row[i], &coo_col[i]);
            coo_row[i]--;  /* adjust from 1-based to 0-based */
            coo_col[i]--;
        }
    }

    if (f !=stdin) fclose(f);

    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);


    int* rowInd1=(int *) calloc(nz, sizeof(int));
    int* colPtr1=(int *) calloc((N+1),sizeof(int));

    int* rowInd2=(int *) calloc(nz, sizeof(int));
    int* colPtr2=(int *) calloc((N+1),sizeof(int));

    coo2csc(rowInd1,colPtr1,coo_row,coo_col,nz,N,0);
    coo2csr(rowInd2,colPtr2,coo_row,coo_col,nz,N,0);

    free(coo_col);
    free(coo_row);


    int b = atoi(argv[2]);

    if(N%b != 0){
      printf("Give another b.");
      exit(1);
    }

    int nb = N/b;

    int wF = atoi(argv[3]); // with F or not 

    // if wF == 0, then the F will be a (nb x nb) array and full of 1. 
    // if wF == 1, then the F will be the array A. 

    if(wF != 1 && wF != 0){
      printf("Error. The last input must be 0 or 1");
      exit(1);
    }

    block* Blocks1=(block*) malloc(nb*nb*sizeof(block));
    for(int i=0; i<nb; i++){
        for(int j=0; j<nb; j++){
            Blocks1[nb*j+i].rowInd = (int*) malloc(sizeof(int));
            Blocks1[nb*j+i].nz=0;
            Blocks1[nb*j+i].count=0;
            Blocks1[nb*j+i].colPtr = (int*) calloc((b+1),sizeof(int));
            Blocks1[nb*j+i].zero = true;   //estw ola midenika
        }
    }
    block* Blocks2=(block*) malloc(nb*nb*sizeof(block));
    for(int i=0; i<nb; i++){
        for(int j=0; j<nb; j++){
            Blocks2[nb*j+i].rowInd = (int*) malloc(sizeof(int));
            Blocks2[nb*j+i].nz=0;
            Blocks2[nb*j+i].count=0;
            Blocks2[nb*j+i].colPtr = (int*) calloc((b+1),sizeof(int));
            Blocks2[nb*j+i].zero = true;
        }
    }

    block* BlocksC=(block*) malloc(nb*nb*sizeof(block));
    for(int i=0; i<nb; i++){
        for(int j=0; j<nb; j++){
            BlocksC[nb*j+i].rowInd = (int*) malloc(sizeof(int));
            BlocksC[nb*j+i].nz=0;
            BlocksC[nb*j+i].count=0;
            BlocksC[nb*j+i].colPtr = (int*) calloc((b+1),sizeof(int));
            for(int k=0; k<b+1; k++){
              BlocksC[nb*j+i].colPtr[k] = 0;
            }
        }
    }

    printf("Blocking.. \n");

    blocking(rowInd1,colPtr1,Blocks1,N,nb,b);
    blocking(rowInd2,colPtr2,Blocks2,N,nb,b);

    free(rowInd1);
    free(colPtr1);
    free(rowInd2);
    free(colPtr2);

    block* BlocksF;

    if(wF == 0){
      int nzF = N*N;
      int* rowIndF = (int*)calloc(nzF,sizeof(int));
      int* colPtrF = (int*)calloc((N+1),sizeof(int));

      createF(rowIndF, colPtrF, N); 

      BlocksF=(block*) malloc(nb*nb*sizeof(block));
      for(int i=0; i<nb; i++){
        for(int j=0; j<nb; j++){
            BlocksF[nb*j+i].rowInd = (int*) malloc(sizeof(int));
            BlocksF[nb*j+i].nz=0;
            BlocksF[nb*j+i].count=0;
            BlocksF[nb*j+i].colPtr = (int*) calloc((b+1),sizeof(int));
            BlocksF[nb*j+i].zero = true;
        }
      }

      blocking(rowIndF,colPtrF,BlocksF,N,nb,b);
    }


    printf("Multiplication..\n");

    clock_gettime(CLOCK_MONOTONIC,&ts_start);
    if(wF == 1){
      maskedMulBlockedArrays(Blocks1,Blocks1,Blocks2,BlocksC,nb,b);
    }else if(wF == 0){
      maskedMulBlockedArrays(BlocksF,Blocks1,Blocks2,BlocksC,nb,b);

      free(BlocksF->colPtr);
      free(BlocksF->rowInd);
      free(BlocksF);
    }

    clock_gettime(CLOCK_MONOTONIC,&ts_end);
    printf("Time for serial imlpementation of %s in secs : %lf \n",argv[1], (double)ts_end.tv_sec +(double)ts_end.tv_nsec*(0.000000001) - (double)ts_start.tv_sec-(double)ts_start.tv_nsec*(0.000000001));


//*****PRINT BlocksC (attention: for nb x nb)*****

    // printf("\n");
    // for(int i=0; i<nb*nb; i++){
    //   printf("rowC[%d] =  \t",i) ;
    //   for(int j=0; j<BlocksC[i].nz;j++){
    //     printf("%d ", BlocksC[i].rowInd[j]);
    //   }
    //   printf("\n");
    // }

    // printf("\n");

    // for(int i=0; i<nb*nb; i++){
    //   printf("colC[%d] =  \t",i ) ;
    //   for(int j=0; j<(b+1);j++){
    //     printf("%d ", BlocksC[i].colPtr[j]);
    //   }
    //   printf("\n");
    // }

    for(int i=0; i<nb*nb; i++){
      if(!Blocks1[i].zero) {
        free(Blocks1[i].rowInd);
        free(Blocks1[i].colPtr);
      }

      if(!Blocks2[i].zero) {
        free(Blocks2[i].rowInd);
        free(Blocks2[i].colPtr);
      }
    }

    free(Blocks1);
    free(Blocks2);
    free(BlocksC->colPtr);
    free(BlocksC->rowInd);
    free(BlocksC);
    return 0;

}
