all: ic_cl

ic_cl: ic_cl.c
	cc -DMAC -arch x86_64 -framework OpenCL -std=c99 -O2 -o ic_cl ic_cl.c -lm
