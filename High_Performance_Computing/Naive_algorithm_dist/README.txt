Compile using:
mpicc Naive_dist.c -lm -o Naive_dist

Run using:
mpirun -np [x] -hostfile myhostfile.txt ./Naive_dist [y]

Where x is the number of ranks, and y is the size of the matrix

example:
mpirun -np 4 --hostfile myhostfile.txt ./Naive_dist 4