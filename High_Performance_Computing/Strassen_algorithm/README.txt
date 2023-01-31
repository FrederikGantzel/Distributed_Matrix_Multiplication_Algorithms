Compile using:
mpicc Strassen.c -lm -o Strassen

Run using:
mpirun -np [x] -hostfile myhostfile.txt ./Strassen [y]

Where x is the number of ranks, and y is the size of the matrix (This algorithm is made to run on only 1 rank,
so the value of x has no effect)

example:
mpirun -np 1 -hostfile myhostfile.txt ./Strassen 10