CC=g++ -std=c++11
CFLAGS= -c -Wall -Wextra -O2
LDFLAGS=

SOURCES= main.cpp
OBJECTS= $(SOURCES:.cpp=.o)

EXECUTABLE= OpticalFlowSegmentation

CFLAGS+= `pkg-config --cflags opencv` 
LDFLAGS+= `pkg-config --libs opencv` 

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm $(OBJECTS) $(EXECUTABLE)
