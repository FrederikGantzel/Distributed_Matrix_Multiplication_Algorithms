Compile using:
mpicc Cannon.c -lm -o Cannon

Run using:
mpirun -np [x] -hostfile myhostfile.txt ./Cannon [y]

Where x is the number of ranks, and y is the size of the matrix

example:
mpirun -np 4 -hostfile myhostfile.txt ./Cannon 4