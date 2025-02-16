CC = g++
SRC = main.cpp
OBJ = main.o 
EXEC = sum_sin
CFLAGS =  

ifeq ($(TYPE), double)
	CFLAGS += -DUSE_DOUBLE
endif

all: $(EXEC)

$(EXEC):$(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(EXEC)

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c $(SRC)

clean:
	rm -f $(OBJ) $(EXEC)

run: $(EXEC)
	./$(EXEC)

