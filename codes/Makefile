SHELL := /bin/bash

CC = gcc -O3
MPICC = mpicc
RM = rm -f

EXECUTABLES = v1 parallel

all: $(EXECUTABLES)

v1: v1.c
	$(CC) $< -o ./$@ 

parallel: parallel.c
	$(MPICC) $< -o ./$@ -lopenblas -lpthread -lm -fopenmp

clean:
	$(RM) *.o *~ $(EXECUTABLES)

default:
	all
