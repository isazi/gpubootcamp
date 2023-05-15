# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

CC := nvc++
CFLAGS := -O3 -w
ACCFLAGS := -fast -acc=gpu -gpu=managed -Minfo=accel

miniWeather: miniWeather_openacc.cpp miniWeather.h
	${CC} ${CFLAGS} ${ACCFLAGS} -o miniWeather miniWeather_openacc.cpp 

clean:
	rm -f *.o miniWeather
