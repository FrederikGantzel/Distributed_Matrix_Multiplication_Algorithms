Compile using:
mpicc Fox.c -lm -o Fox

Run using:
mpirun -np [x] -hostfile myhostfile.txt ./Fox [y]

Where x is the number of ranks, and y is the size of the matrix

example:
mpirun -np 4 -hostfile myhostfile.txt ./Fox 4