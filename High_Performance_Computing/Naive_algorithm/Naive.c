#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>


int main(int argc, char **argv) {

  int my_rank, nprocs;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  int n; //matrix size
  sscanf(argv[1],"%d",&n);

  if (argc != 2) {
    fprintf(stderr,"Please provide the matrix size.\n");
    MPI_Finalize();
    exit(0);
  }

  int ** matrixA=(int**)malloc(sizeof(int*)*n);
  int ** matrixB=(int**)malloc(sizeof(int*)*n);
  int ** matrixC=(int**)malloc(sizeof(int*)*n);

  int top_rank = nprocs-1;
  int leftovers = n % nprocs;
  int subSize = (n-leftovers)/nprocs;
  MPI_Status status;

  int * rowRange=(int*)malloc(sizeof(int)*n);
  int * rowRange_split=(int*)malloc(sizeof(int)*subSize);

  //create test matrces, matrix A has all 1's, matrix B has all 2's, matrix C is the resulting matrix
  for (int i=0; i<n; i++) {
    matrixA[i]=(int*)malloc(sizeof(int)*n);
    matrixB[i]=(int*)malloc(sizeof(int)*n);
    matrixC[i]=(int*)malloc(sizeof(int)*n);
    for (int j=0; j<n; j++) {
      matrixA[i][j] = 1.0;
      matrixB[i][j] = 2.0;
      matrixC[i][j] = 0.0;
    }
    rowRange[i]=i;
  }

  MPI_Scatter(rowRange, subSize, MPI_INT, rowRange_split, subSize, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  double multiplicationStart=MPI_Wtime();

  for(int i=0; i<subSize; i++) { //compute matrix
    for(int j=0; j<n; j++) {
      for (int k=0; k<n; k++) {
        matrixC[rowRange_split[i]][j] = matrixC[rowRange_split[i]][j] + (matrixA[rowRange_split[i]][k] * matrixB[k][j]);
      }
      if (my_rank != 0) {
        MPI_Send(&matrixC[rowRange_split[i]][j], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      }
      else {
        for (int x=1; x<nprocs; x++) {
          MPI_Recv(&matrixC[(x*subSize)+i][j], 1, MPI_INT, x, 0, MPI_COMM_WORLD, &status);
        }
      }
    }
  }

  if(leftovers != 0){ //compute leftovers
    if(my_rank == top_rank) {
      for (int i=(n-leftovers); i<n; i++) {
        for (int j=0; j<n; j++) {
          for (int k=0; k<n; k++) {
            matrixC[i][j] = matrixC[i][j] + (matrixA[i][k] * matrixB[k][j]);
          }
          MPI_Send(&matrixC[i][j], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
      }
    }
    if(my_rank == 0) {
      for (int i=(n-leftovers); i<n; i++) {
        for (int j=0; j<n; j++) {
          MPI_Recv(&matrixC[i][j], 1, MPI_INT, top_rank, 0, MPI_COMM_WORLD, &status);
        }
      }
    }
  }

  double multiplicationEnd=MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  //PRINTING EVERYTHING

  if (my_rank == 0) {
    printf("\nMultiplying %d x %d matrix A, which contains all 1's, and %d x %d matrix B; which contains all 2's\n", n, n, n, n);

    for (int i=0; i<n; i++) { //prints the resulting matrix, do not do this for large matrix sizes
      for (int j=0; j<n; j++) {
        printf("%d ", matrixC[i][j]);
      }
      printf("\n");
    }

    printf("Total time to multiply: %f (s)\n\n", multiplicationEnd - multiplicationStart);
  }


  MPI_Finalize();
  return 0;
}
