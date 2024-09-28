CC := gcc
CFLAGS := -Wall -Wextra -lm
DEBUG_FLAGS := -g -fsanitize=address
SOURCES = $(wildcard src/*.c)
OBJS = $(SOURCES:.c=.o)

TEST_SOURCES = $(wildcard tests/*.c)
TEST_OBJS = $(TEST_SOURCES:.c=.o)

all: neural

#Build
neural : $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lm


%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<


run_tests: $(OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ src/perceptron.o src/train.c $(TEST_OBJS) -lcriterion -lm


debug: CFLAGS += $(DEBUG_FLAGS)
debug: all



.PHONY: clean
clean:
	$(RM) $(OBJS) $(TARGET) $(TEST_OBJS) neural run_tests

.PHONY: test
test: run_tests
	./run_tests
