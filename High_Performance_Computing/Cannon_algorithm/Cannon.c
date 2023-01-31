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

  for (int i=0; i<LMS; i++) {
    matrixA[i]=(int*)malloc(sizeof(int)*LMS);
    matrixB[i]=(int*)malloc(sizeof(int)*LMS);
    matrixC[i]=(int*)malloc(sizeof(int)*LMS);
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


  //first, we shift blocks in row i on the A matrix i spots to the left, and blocks in column j on the B matrix j spots up

  int posi = my_rank/SP;
  int posj = my_rank%SP;

  int AdestSendRank, AdestRecvRank, BdestSendRank, BdestRecvRank;

  //determine what rank each rank sends to, and what rank it recieves from
  if (posj-posi < 0) {
    AdestSendRank = rankMatrix[posi][posj-posi+SP];
  }
  else {
    AdestSendRank = rankMatrix[posi][posj-posi];
  }

  if (posj+posi >= SP) {
    AdestRecvRank = rankMatrix[posi][posj+posi-SP];
  }
  else {
    AdestRecvRank = rankMatrix[posi][posj+posi];
  }

  if (posi-posj < 0) {
    BdestSendRank = rankMatrix[posi-posj+SP][posj];
  }
  else {
    BdestSendRank = rankMatrix[posi-posj][posj];
  }

  if (posi+posj >= SP) {
    BdestRecvRank = rankMatrix[posi+posj-SP][posj];
  }
  else {
    BdestRecvRank = rankMatrix[posi+posj][posj];
  }

  int ** tempMatrixA=(int**)malloc(sizeof(int*)*LMS);
  int ** tempMatrixB=(int**)malloc(sizeof(int*)*LMS);

  MPI_Status status;
  MPI_Request request;

  MPI_Barrier(MPI_COMM_WORLD);

  //shift each block the appropriate amount
  for (int i=0; i<LMS; i++) {
    tempMatrixA[i]=(int*)malloc(sizeof(int)*LMS);
    tempMatrixB[i]=(int*)malloc(sizeof(int)*LMS);
    for (int j=0; j<LMS; j++) {
      tempMatrixA[i][j] = matrixA[i][j];
      tempMatrixB[i][j] = matrixB[i][j];

      MPI_Isend(&tempMatrixA[i][j], 1, MPI_INT, AdestSendRank, 0, MPI_COMM_WORLD, &request);
      MPI_Isend(&tempMatrixB[i][j], 1, MPI_INT, BdestSendRank, 1, MPI_COMM_WORLD, &request);

      MPI_Recv(&matrixA[i][j], 1, MPI_INT, AdestRecvRank, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&matrixB[i][j], 1, MPI_INT, BdestRecvRank, 1, MPI_COMM_WORLD, &status);

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }


  //from now on, the rows and columns only shift 1 spot at a time
  if (posj-1 < 0) {
    AdestSendRank = rankMatrix[posi][posj-1+SP];
  }
  else {
    AdestSendRank = rankMatrix[posi][posj-1];
  }

  if (posj+1 >= SP) {
    AdestRecvRank = rankMatrix[posi][posj+1-SP];
  }
  else {
    AdestRecvRank = rankMatrix[posi][posj+1];
  }

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
  for (int i=0; i<LMS; i++) {
    for (int j=0; j<LMS; j++) {
      for (int k=0; k<SP; k++) {
        //calculate part of matrix C from the local matrix A and B
        for (int z=0; z<LMS; z++) {
          matrixC[i][j] = matrixC[i][j] + (matrixA[i][z]*matrixB[z][j]);
        }

        //we switch the blocks around
        for (int x=0; x<LMS; x++) {
          for (int y=0; y<LMS; y++) {
            tempMatrixA[x][y] = matrixA[x][y];
            tempMatrixB[x][y] = matrixB[x][y];

            MPI_Isend(&tempMatrixA[x][y], 1, MPI_INT, AdestSendRank, 0, MPI_COMM_WORLD, &request);
            MPI_Isend(&tempMatrixB[x][y], 1, MPI_INT, BdestSendRank, 1, MPI_COMM_WORLD, &request);

            MPI_Recv(&matrixA[x][y], 1, MPI_INT, AdestRecvRank, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrixB[x][y], 1, MPI_INT, BdestRecvRank, 1, MPI_COMM_WORLD, &status);

            MPI_Barrier(MPI_COMM_WORLD);
          }
        }

      }
    }
  }


  //printing for verification

  if (my_rank == 0) {
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
