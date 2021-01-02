all:	dchex.o
	mex -O choluprk1.c -lblas
	mex -O choldnrk1.c -lblas
	mex -O cholupexch.c dchex.o

dchex.o:	dchex.f
	g77 dchex.f -c -funroll-all-loops -fno-f2c -O3

clean:
	rm -f *.o *.mexglx *~ \#*
