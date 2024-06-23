CC=gcc
CFLAGS=-I. -lgsl -lgslcblas -lm -g -Wall
DEPS = algorithms.h
OBJ = main.o algorithms.o 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
