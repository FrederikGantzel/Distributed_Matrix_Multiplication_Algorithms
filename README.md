# Distributed_Matrix_Multiplication_Algorithms
Final project for the course CS599 - "High Performance Computing", NAU Spring Semester 2021

This is probably my favorite course I took at NAU, and this final project is a big part of what made it so enjoyable. Big thanks to Dr. Gowanlock for some great assignments.

## Installation
Download the "High_Performance_Computing" folder. To run a program, set the directory to the location of that specific program, and refer to the README file of that program for instructions about how to compile and run.

## Usage
Provides 5 different algorithms for matrix multiplication:

- __The Naive Algorithm__

The classic Naive Algorithm for matrix multiplication. It does not have distributed memory or anything special


- __Distributed Memory Version of the Naive Algorithm__

The Naive Algorithm with added distributed memory functionality. The algorithm should run faster by being spread over multiple ranks, but it is very poorly optimized by virtue of being just the regular old Naive Algorithm

- __Fox Algorithm__

The Fox Algorithm uses distributed memory and rotates "blocks" of information between the ranks to speed up computation.

- __Cannon Algorithm__

The Cannon Algorithm works much like the Fox algorithm, but rotates the blocks of information in a different way. Refer to the project report for a more detailed description of the Fox and Cannon algorithms

- __Strassen Algorithm__

Does not use distributed memory, but rather uses a mathematical trick to recude the number of calculations needed for matrix multiplication. This is kind of a "bonus" algorithm, and it is not relevant to, nor do I even mention it in the project report.

## Some technical stuff
As the focuses of the programs are the algoror all the algorithms themselves, the matrices being multiplied are gerenated in the program (matrix A containing all 1's, and matrix B containing all 2's, and thus none of the programs take a matrix as input.

For verification, the distributed mempory algorithms output just a single block of the resulting matrix, while the non-distributed memory algorithms output the whole resulting matrix.

All the programs (including the non-distributed memory ones) are compiled with mpicc (which is the reason for the hostfiles), to simulate the multiple ranks. I'm sure it's very simple to convert the non-distributed memory algorithms here to be compiled using just gcc or something, but I'm too lazy to figure it out. I ran all the programs using the Ubuntu Linux virtual machine, and was able to easily install mpicc with the command __sudo apt install mpich__, but I'm not sure how to install it with other systems.

The programs get very slow very fast for larger amtrix sizes. The sizes I ran with in the project report was run on NAU's "Monsoon" supercomputer, so don't try to run with matrices of those sizes on your laptop.

## Example runs
![image](https://user-images.githubusercontent.com/91853323/215654580-d37871c0-f905-42c5-a4fc-47fb2b3dde54.png)

![image](https://user-images.githubusercontent.com/91853323/215654691-c9d068c4-0264-488d-9df5-edfb2580aa37.png)

![image](https://user-images.githubusercontent.com/91853323/215654823-89162845-fb06-47e1-bd54-148ebdf6a45f.png)




