#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "mmio.c"
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>
#include <mpi.h>
#include <omp.h>
#include <cblas.h>

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

void transpose(block *array, int nb){

    block* new_array = (block*)malloc(nb*nb*sizeof(block));

    for (int i = 0; i < nb; ++i ){
       for (int j = 0; j < nb; ++j ){
          // Index in the original matrix.
          int index1 = i*nb+j;

          // Index in the transpose matrix.
          int index2 = j*nb+i;

          new_array[index2] = array[index1];
       }
    }

    for (int i=0; i<nb*nb; i++) {
        array[i] = new_array[i];
    }

    free(new_array);
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
    if(Blocks[i].zero == true){
      free(Blocks[i].rowInd);
      //free(Blocks[i].colPtr);
    }
  }
}



void maskedMultiply(block F, bool* filter, block A, block B, block* C, int n, bool firstTime, int index){
  int size = 0;
  if(firstTime){
    #pragma omp parallel for
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
    #pragma omp parallel for
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

int main(int argc, char *argv[]) {
     struct timespec ts_start,ts_end;

    // MPI
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);    

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int b=20; 

    //create a type for struct block 
    const int nitems=5;
    int          blocklengths[5] = {(b*b),(b+1),1,1,1};
    MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_C_BOOL};
    MPI_Datatype mpi_block;
    MPI_Aint     offsets[5];

    offsets[0] = offsetof(block, rowInd);
    offsets[1] = offsetof(block, colPtr);
    offsets[2] = offsetof(block, nz);
    offsets[3] = offsetof(block, count);
    offsets[4] = offsetof(block, zero);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_block);
    MPI_Type_commit(&mpi_block);

 
    int nb, chunks, N;
    int index_col;
    block* Block1_i;
    block* Block2_i;
    block* Block2_i_send;
    block* Block2_i_received;
    block* C_final;

    clock_gettime(CLOCK_MONOTONIC,&ts_start);

    if(world_rank == 0){

      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, nz, i,j,k; 
      int *coo_col, *coo_row;

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
  
      nb = N/b;
      chunks = nb/p;

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

      blocking(rowInd1,colPtr1,Blocks1,N,nb,b);   
      blocking(rowInd2,colPtr2,Blocks2,N,nb,b);

      transpose(Blocks1,nb);

      free(rowInd1);
      free(colPtr1);
      free(rowInd2);
      free(colPtr2);

      Block1_i = (block*)malloc(chunks*nb*sizeof(block));
      Block2_i_send = (block*)malloc(chunks*nb*sizeof(block));
      
      if(p>1){
        //Broadcast to all other processes
        for(int i = 1; i < p; i++) {
          // MPI_Send to each process the appropriate array
          MPI_Send(&N, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
          MPI_Send(&Blocks1[i * chunks * nb], chunks * nb, mpi_block, i, 0, MPI_COMM_WORLD);
          MPI_Send(&Blocks2[i * chunks * nb], chunks * nb, mpi_block, i, 0, MPI_COMM_WORLD);
          index_col = i;
          MPI_Send(&index_col, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
      }

      index_col = 0;
      memcpy(Block1_i, Blocks1, chunks * nb * sizeof(block));
      memcpy(Block2_i_send, Blocks2, chunks * nb * sizeof(block));

      free(Blocks1->colPtr);
      free(Blocks1->rowInd);
      free(Blocks1);
      free(Blocks2->colPtr);
      free(Blocks2->rowInd);
      free(Blocks2);

      C_final=(block*) malloc(chunks*nb*sizeof(block));
      for(int i=0; i<chunks; i++){
        for(int j=0; j<nb; j++){
          C_final[nb*i+j].rowInd = (int*) malloc(sizeof(int));
          C_final[nb*i+j].nz=0;
          C_final[nb*i+j].count=0;
          C_final[nb*i+j].colPtr = (int*) calloc((b+1),sizeof(int));
          for(int k=0; k<b+1; k++){
            C_final[nb*i+j].colPtr[k] = 0;
          }
        }
      }

    }else{
      MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      nb = N/b;
      chunks = nb/p;

      Block1_i = (block*)malloc(chunks*nb*sizeof(block));
      Block2_i_send = (block*)malloc(chunks*nb*sizeof(block));

      // First receive from the mother process
      MPI_Recv(Block1_i, chunks*nb, mpi_block, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(Block2_i_send, chunks*nb, mpi_block, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&index_col, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }   

    //for the results of each process
    block* BlocksC=(block*) malloc(chunks*chunks*sizeof(block));
    for(int i=0; i<chunks; i++){
      for(int j=0; j<chunks; j++){
        BlocksC[nb*i+j].rowInd = (int*) malloc(sizeof(int));
        BlocksC[nb*i+j].nz=0;
        BlocksC[nb*i+j].count=0;
        BlocksC[nb*i+j].colPtr = (int*) calloc((b+1),sizeof(int));
        for(int k=0; k<b+1; k++){
          BlocksC[nb*i+j].colPtr[k] = 0;
        }
      }
    }

    for(int i = 0; i < (p-1); i ++) {

      bool* filter = (bool*)malloc(b*b*sizeof(bool));

      for(int r=0; r<chunks; r++){
        for(int c=0; c<chunks; c++){
          bool firstTime = true;

          for(int f=0; f<b*b; f++){
            filter[f] = true;
          }

          for(int k=0; k<nb; k++){
            if(Block1_i[nb*r+k].zero == false && Block2_i_send[nb*c+k].zero == false){
              maskedMultiply(Block1_i[nb*r+c],filter,Block1_i[nb*r+k],Block2_i_send[nb*c+k],&BlocksC[nb*r+c], b, firstTime,(nb*r+c));
              firstTime = false;
            }
          }
        }
      }

      free(filter);

      if(world_rank != 0){

        MPI_Send(BlocksC, chunks * chunks, mpi_block, 0, i, MPI_COMM_WORLD);
        MPI_Send(&index_col, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

      }else{

        for(int bc=0; bc<chunks; bc++){
          for(int br=0; br<chunks; br++){
            C_final[chunks*index_col+chunks*bc+br] = BlocksC[chunks*bc+br];
          }
        }

        for(int i=1; i<p; i++){

          block* BlocksC_received=(block*) malloc(chunks*chunks*sizeof(block));
          for(int i=0; i<chunks; i++){
            for(int j=0; j<chunks; j++){
              BlocksC_received[nb*i+j].rowInd = (int*) malloc(sizeof(int));
              BlocksC_received[nb*i+j].nz=0;
              BlocksC_received[nb*i+j].count=0;
              BlocksC_received[nb*i+j].colPtr = (int*) calloc((b+1),sizeof(int));
              for(int k=0; k<b+1; k++){
                BlocksC_received[nb*i+j].colPtr[k] = 0;
              }
            }
          }

          MPI_Recv(BlocksC_received, chunks * chunks, mpi_block, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&index_col, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          
          C_final = (block*)realloc(C_final, (chunks*nb+chunks*chunks)*sizeof(block));

          for(int bc=0; bc<chunks; bc++){
            for(int br=0; br<chunks; br++){
              C_final[chunks*index_col+i+chunks*bc+br] = BlocksC_received[chunks*bc+br];
            }
          }

          free(BlocksC_received->rowInd);
          free(BlocksC_received->colPtr);
          free(BlocksC_received);
         
        }
      }

      //send the Block2_i_send to the next process
      MPI_Send(Block2_i_send, chunks * nb, mpi_block, (world_rank + 1) % p, i, MPI_COMM_WORLD);
      MPI_Send(&index_col, 1, MPI_INT, (world_rank + 1) % p, i, MPI_COMM_WORLD);


      if(world_rank != 0){
        Block2_i_received = (block*)malloc(chunks*nb*sizeof(block));
        MPI_Recv(Block2_i_received, chunks * nb, mpi_block, world_rank - 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&index_col, 1, MPI_INT, world_rank - 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }else{
        Block2_i_received = (block*)malloc(chunks*nb*sizeof(block));
        MPI_Recv(Block2_i_received, chunks * nb, mpi_block, p - 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&index_col, 1, MPI_INT, p-1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      memcpy(Block2_i_send, Block2_i_received, chunks * nb * sizeof(block));

      if(i==p-1){
        free(C_final->colPtr);
        free(C_final->rowInd);
        free(C_final);
      }

    }

    clock_gettime(CLOCK_MONOTONIC,&ts_end);
    printf("Time for serial imlpementation in secs : %lf \n",(double)ts_end.tv_sec +(double)ts_end.tv_nsec*(0.000000001) - (double)ts_start.tv_sec-(double)ts_start.tv_nsec*(0.000000001));

     
    free(Block1_i->rowInd);
    free(Block1_i->colPtr);
    free(Block1_i);
    free(Block2_i->rowInd);
    free(Block2_i->colPtr);
    free(Block2_i);
    free(Block2_i_received->rowInd);
    free(Block2_i_received->colPtr);
    free(Block2_i_received);
    free(Block2_i_send->colPtr);
    free(Block2_i_send->rowInd);
    free(Block2_i_send);
    free(BlocksC->colPtr);
    free(BlocksC->rowInd);
    free(BlocksC);

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;

}