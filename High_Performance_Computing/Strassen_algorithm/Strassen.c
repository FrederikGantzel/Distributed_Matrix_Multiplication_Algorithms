#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>


// echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
// mpicc Strassen.c -lm -o Strassen
// mpirun -np 1 -hostfile myhostfile.txt ./Strassen 10

int *** matrixSplit(int ** matr, int n);
int ** matrixAdd(int ** matrixA, int ** matrixB, int n);
int ** matrixSubtract(int ** matrixA, int ** matrixB, int n);
int ** strassen(int ** matrix1, int ** matrix2, int size);


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
  int ** matrixC;


  //create test matrces, matrix A has all 1's, matrix B has all 2's, matrix C is the resulting matrix
  for (int i=0; i<n; i++) {
    matrixA[i]=(int*)malloc(sizeof(int)*n);
    matrixB[i]=(int*)malloc(sizeof(int)*n);
    for (int j=0; j<n; j++) {
      matrixA[i][j] = 1;
      matrixB[i][j] = 2;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  double multiplicationStart=MPI_Wtime();

  matrixC = strassen(matrixA, matrixB, n);

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

int *** matrixSplit(int ** matr, int n) {
  int newn = (n/2);
  int *** resMatrix=(int***)malloc(sizeof(int**)*4);


  resMatrix[0]=(int**)malloc(sizeof(int*)*newn);
  for (int i=0; i<newn; i++) {
    resMatrix[0][i]=(int*)malloc(sizeof(int)*newn);
    for (int j=0; j<newn; j++) {
      resMatrix[0][i][j] = matr[i][j];
    }
  }

  resMatrix[1]=(int**)malloc(sizeof(int*)*newn);
  for (int i=0; i<newn; i++) {
    resMatrix[1][i]=(int*)malloc(sizeof(int)*newn);
    for (int j=0; j<newn; j++) {
      resMatrix[1][i][j] = matr[i][j+newn];
    }
  }

  resMatrix[2]=(int**)malloc(sizeof(int*)*newn);
  for (int i=0; i<newn; i++) {
    resMatrix[2][i]=(int*)malloc(sizeof(int)*newn);
    for (int j=0; j<newn; j++) {
      resMatrix[2][i][j] = matr[i+newn][j];
    }
  }

  resMatrix[3]=(int**)malloc(sizeof(int*)*newn);
  for (int i=0; i<newn; i++) {
    resMatrix[3][i]=(int*)malloc(sizeof(int)*newn);
    for (int j=0; j<newn; j++) {
      resMatrix[3][i][j] = matr[i+newn][j+newn];
    }
  }

  return resMatrix;
}

int ** matrixAdd(int ** matrixA, int ** matrixB, int n) { //adds matrixA and matrixB
  int ** matrixC=(int**)malloc(sizeof(int*)*n);
  for (int i=0; i<n; i++) {
    matrixC[i]=(int*)malloc(sizeof(int)*n);
    for (int j=0; j<n; j++) {
      matrixC[i][j] = matrixA[i][j]+matrixB[i][j];
    }
  }

  return matrixC;
}

int ** matrixSubtract(int ** matrixA, int ** matrixB, int n) { //subtracts matrixB from matrixA
  int ** matrixC=(int**)malloc(sizeof(int*)*n);
  for (int i=0; i<n; i++) {
    matrixC[i]=(int*)malloc(sizeof(int)*n);
    for (int j=0; j<n; j++) {
      matrixC[i][j] = matrixA[i][j]-matrixB[i][j];
    }
  }

  return matrixC;
}

int ** strassen(int ** matrix1, int ** matrix2, int size) {

  int ** matrixA;
  int ** matrixB;
  int n = size;
  bool addedZeros = false;
  //if matrix is 1x1
  if (n == 1) {
    int ** matrixC=(int**)malloc(sizeof(int*)*1);
    matrixC[0]=(int*)malloc(sizeof(int)*1);
    matrixC[0][0] = matrix1[0][0] * matrix2[0][0];
    return matrixC;
  }

  if ((n%2) != 0) { //if not divisible by two, add an extra row and columns with 0's
    matrixA=(int**)malloc(sizeof(int*)*(n+1));
    matrixB=(int**)malloc(sizeof(int*)*(n+1));

    for (int i=0; i<n; i++) {
      matrixA[i]=(int*)malloc(sizeof(int)*(n+1));
      matrixB[i]=(int*)malloc(sizeof(int)*(n+1));
      for (int j=0; j<n; j++) {
        matrixA[i][j] = matrix1[i][j];
        matrixB[i][j] = matrix2[i][j];
      }
    }
    matrixA[n]=(int*)malloc(sizeof(int)*(n+1));
    matrixB[n]=(int*)malloc(sizeof(int)*(n+1));
    for (int i=0; i<n+1; i++) {
      matrixA[i][n] = 0;
      matrixA[n][i] = 0;
      matrixB[i][n] = 0;
      matrixB[n][i] = 0;
    }
    n = n+1;
    addedZeros = true;
  }
  else {
    matrixA = matrix1;
    matrixB = matrix2;
  }

  int *** splitA = matrixSplit(matrixA, n);
  int *** splitB = matrixSplit(matrixB, n);

  int halfn = (n/2);

  int ** a = splitA[0];
  int ** b = splitA[1];
  int ** c = splitA[2];
  int ** d = splitA[3];
  int ** e = splitB[0];
  int ** f = splitB[1];
  int ** g = splitB[2];
  int ** h = splitB[3];

  int ** p1 = strassen(a, matrixSubtract(f, h, halfn), halfn);
  int ** p2 = strassen(matrixAdd(a, b, halfn), h, halfn);
  int ** p3 = strassen(matrixAdd(c, d, halfn), e, halfn);
  int ** p4 = strassen(d, matrixSubtract(g, e, halfn), halfn);
  int ** p5 = strassen(matrixAdd(a, d, halfn), matrixAdd(e, h, halfn), halfn);
  int ** p6 = strassen(matrixSubtract(b, d, halfn), matrixAdd(g, h, halfn), halfn);
  int ** p7 = strassen(matrixSubtract(a, c, halfn), matrixAdd(e, f, halfn), halfn);

  int ** c11 = matrixAdd(matrixSubtract(matrixAdd(p5, p4, halfn), p2, halfn), p6, halfn);
  int ** c12 = matrixAdd(p1, p2, halfn);
  int ** c21 = matrixAdd(p3, p4, halfn);
  int ** c22 = matrixSubtract(matrixSubtract(matrixAdd(p1, p5, halfn), p3, halfn), p7, halfn);

  if (addedZeros == true) {
    n = n-1;
  }

  int ** matrixC=(int**)malloc(sizeof(int*)*n);
  for (int i=0; i<halfn; i++) {
    matrixC[i]=(int*)malloc(sizeof(int)*n);
    for (int j=0; j<n; j++) {
      if (j<halfn) {
        matrixC[i][j] = c11[i][j];
      }
      else {
        matrixC[i][j] = c12[i][j-halfn];
      }
    }
  }
  for (int i=halfn; i<n; i++) {
    matrixC[i]=(int*)malloc(sizeof(int)*n);
    for (int j=0; j<n; j++) {
      if (j<halfn) {
        matrixC[i][j] = c21[i-halfn][j];
      }
      else {
        matrixC[i][j] = c22[i-halfn][j-halfn];
      }
    }
  }

  return matrixC;

}
