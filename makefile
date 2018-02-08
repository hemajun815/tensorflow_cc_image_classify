target = tfcc_mnist.out
cc = g++ -std=c++11
include = -I/usr/local/tensorflow/include
lib = -L/usr/local/tensorflow/lib -ltensorflow_framework -ltensorflow_cc
flag = -Wl,-rpath=/usr/local/tensorflow/lib
source = ./src/dataset.cc ./src/image_classifier.cc ./src/main.cc

$(target): $(source)
	$(cc) $(source) -o $(target) $(include) $(lib) $(flag)

clean:
	rm $(target)

run: $(target)
	./$(target)