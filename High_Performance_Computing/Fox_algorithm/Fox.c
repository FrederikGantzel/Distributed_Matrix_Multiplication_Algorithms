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
    if (my_rank == 0) {
      fprintf(stderr,"Please provide the matrix size.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  int SP = sqrt(nprocs); //square root of p
  if ((SP*SP) != nprocs) {
    if (my_rank == 0) {
      printf("Number of ranks must have an integer square root. (1, 4, 9, 16, 25, etc)\n");
    }
    MPI_Finalize();
    exit(0);
  }
  if (n % SP != 0) {
    if (my_rank == 0) {
      printf("Square root of p must evenly divide n\n");
    }
    MPI_Finalize();
    exit(0);
  }
  int LMS = (n/SP); //local matrix size


  //create test matrces, matrix A has all 1's, matrix B has all 2's, matrix C is the resulting matrix
  //we assume that matrixes A and B have already beed distributed to their corresponding ranks
  int ** matrixA=(int**)malloc(sizeof(int*)*LMS);
  int ** matrixB=(int**)malloc(sizeof(int*)*LMS);
  int ** matrixC=(int**)malloc(sizeof(int*)*LMS);
  int ** tempMatrixA=(int**)malloc(sizeof(int*)*LMS);
  int ** tempMatrixB=(int**)malloc(sizeof(int*)*LMS);

  for (int i=0; i<LMS; i++) {
    matrixA[i]=(int*)malloc(sizeof(int)*LMS);
    matrixB[i]=(int*)malloc(sizeof(int)*LMS);
    matrixC[i]=(int*)malloc(sizeof(int)*LMS);
    tempMatrixA[i]=(int*)malloc(sizeof(int)*LMS);
    tempMatrixB[i]=(int*)malloc(sizeof(int)*LMS);
    for (int j=0; j<LMS; j++) {
      matrixA[i][j] = 1;
      matrixB[i][j] = 2;
      matrixC[i][j] = 0;
    }
  }

  //create a matrix representing the blocks that A and B are split into, and the ranks they are assigned to
  int ** rankMatrix=(int**)malloc(sizeof(int*)*SP);
  for (int i=0; i<SP; i++) {
    rankMatrix[i]=(int*)malloc(sizeof(int)*SP);
    for (int j=0; j<SP; j++) {
      rankMatrix[i][j] = (i*SP)+j;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Status status;
  MPI_Request request;

  int posi = my_rank/SP;
  int posj = my_rank%SP;

  int BdestSendRank, BdestRecvRank;


  //each phase, the columns shift 1 spot up at a time

  if (posi-1 < 0) {
    BdestSendRank = rankMatrix[posi-1+SP][posj];
  }
  else {
    BdestSendRank = rankMatrix[posi-1][posj];
  }

  if (posi+1 >= SP) {
    BdestRecvRank = rankMatrix[posi+1-SP][posj];
  }
  else {
    BdestRecvRank = rankMatrix[posi+1][posj];
  }

  MPI_Barrier(MPI_COMM_WORLD);
  //now we perform the matrix multiplication

  //we broadcast the initial block of a across the rows
  int AbroadcastingRank = rankMatrix[posi][posi];
  for (int x=0; x<LMS; x++) {
    for (int y=0; y<LMS; y++) {

      if (my_rank == AbroadcastingRank) {
        for (int z=0; z<SP; z++) {
          MPI_Isend(&matrixA[x][y], 1, MPI_INT, rankMatrix[posi][z], 0, MPI_COMM_WORLD, &request);
        }
      }

      MPI_Recv(&tempMatrixA[x][y], 1, MPI_INT, AbroadcastingRank, 0, MPI_COMM_WORLD, &status);

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  for (int k=1; k<=SP; k++) {

    //calculate value of C from the local matrixes tempMatrix A and matrixB
    for (int i=0; i<LMS; i++) {
      for (int j=0; j<LMS; j++) {
        for (int z=0; z<LMS; z++) {
          matrixC[i][j] = matrixC[i][j] + (tempMatrixA[i][z]*matrixB[z][j]);
        }
      }
    }

    //broadcast the next block of matrix A, and shift blocks of matrix B one spot up
    if (k<SP) {
      AbroadcastingRank = rankMatrix[posi][(posi+k)%SP];
      for (int x=0; x<LMS; x++) {
        for (int y=0; y<LMS; y++) {
          tempMatrixB[x][y] = matrixB[x][y];

          MPI_Isend(&tempMatrixB[x][y], 1, MPI_INT, BdestSendRank, 1, MPI_COMM_WORLD, &request);
          if (my_rank == AbroadcastingRank) {
            for (int z=0; z<SP; z++) {
              MPI_Isend(&matrixA[x][y], 1, MPI_INT, rankMatrix[posi][z], 0, MPI_COMM_WORLD, &request);
            }
          }

          MPI_Recv(&tempMatrixA[x][y], 1, MPI_INT, AbroadcastingRank, 0, MPI_COMM_WORLD, &status);
          MPI_Recv(&matrixB[x][y], 1, MPI_INT, BdestRecvRank, 1, MPI_COMM_WORLD, &status);

          MPI_Barrier(MPI_COMM_WORLD);
        }
      }
    }

  }


  //printing for verification

  if (my_rank == 7) {
    printf("Rank %d has:\n", my_rank);
    for (int i=0; i<LMS; i++) {
      for (int j=0; j<LMS; j++) {
        printf("%d ", matrixC[i][j]);
      }
      printf("\n");
    }
  }


  MPI_Finalize();
  return 0;
}
