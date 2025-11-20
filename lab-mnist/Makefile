NVCXX = nvc++
NVCXXFLAGS = -O0 -std=c++14 -tp=cascadelake -fast -acc -gpu=cc60 -Minfo=accel
MNIST=-I/share/testcases/lab-mnist/weights/mnist/include
TARGETS = mnist

.PHONY: all
all: $(TARGETS)

%: %.cpp
	$(NVCXX) $(NVCXXFLAGS) $(MNIST) -o $@ $<

clean:
	rm -f $(TARGETS)
