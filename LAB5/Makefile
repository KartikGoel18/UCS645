CC = mpicc
CFLAGS = -Wall -O2
LDLIBS = -lm

all: q1_daxpy q2_bcast q3_dotprod q4_primes q5_perfect

q1_daxpy: q1_daxpy.c
	$(CC) $(CFLAGS) -o q1_daxpy q1_daxpy.c $(LDLIBS)

q2_bcast: q2_bcast.c
	$(CC) $(CFLAGS) -o q2_bcast q2_bcast.c $(LDLIBS)

q3_dotprod: q3_dotprod.c
	$(CC) $(CFLAGS) -o q3_dotprod q3_dotprod.c $(LDLIBS)

q4_primes: q4_primes.c
	$(CC) $(CFLAGS) -o q4_primes q4_primes.c $(LDLIBS)

q5_perfect: q5_perfect.c
	$(CC) $(CFLAGS) -o q5_perfect q5_perfect.c $(LDLIBS)

clean:
	rm -f q1_daxpy q2_bcast q3_dotprod q4_primes q5_perfect
