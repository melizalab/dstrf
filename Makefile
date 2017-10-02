PYTHON_PREFIX=$(shell python-config --prefix)
PYTHON_CFLAGS=$(shell python-config --cflags)
PYTHON_LIBS=$(shell python-config --libs)

TARGET = cneurons

all: $(TARGET).so

$(TARGET).so: $(TARGET).o
	 g++ -O2 --std=c++11 -shared -Wl,--export-dynamic $(TARGET).o -lboost_python -L$(PYTHON_PREFIX)/lib $(PYTHON_LIBS) -o $(TARGET).so


$(TARGET).o: $(TARGET).cpp
	 g++ -O2 --std=c++11 $(PYTHON_CFLAGS) -fPIC -c $(TARGET).cpp

clean:
	rm -f $(TARGET).so $(TARGET).o
