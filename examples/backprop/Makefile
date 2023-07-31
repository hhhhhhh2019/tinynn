LIBS = ''

CC_FLAGS = -I ../../src -c
LD_FLAGS = -lm

ifeq ($(strip $(LIBS)),)
	CC_FLAGS += `pkg-config --cflags $(LIBS)`
	LD_FALGS += `pkg-config --libs $(LIBS)`
endif

ifdef DEBUG
	CC_FLAGS += -fsanitize=address -g -DDEBUG
	LD_FLAGS += -fsanitize=address -g -lasan
endif

CC ?= gcc
LD = $(CC)


SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)
OBJECTS += ../../bin/tinynn.o


%.o: %.c
	$(CC) $(CC_FLAGS) $< -o $@

all: $(OBJECTS)
	$(LD) $^ $(LD_FLAGS) -o ../../bin/backprop


clean:
	rm *.o -f
