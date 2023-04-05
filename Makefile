src = $(wildcard *.cpp)
obj = $(src:.c=.o)
LDFLAGS = -lsfml-window -lsfml-graphics -lsfml-system

nn: $(obj)
	g++ -o $@ $^ $(LDFLAGS)
.PHONY: clean
clean:
	rm -f $(obj) nn
