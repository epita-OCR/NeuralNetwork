CC := gcc
CFLAGS := -Wall -Wextra -lm
TARGET = neural
OBJS = src/perceptron.o src/train.o src/main.o 

$(TARGET): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(TARGET)
