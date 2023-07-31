all: src examples

mkdirs:
	mkdir -p bin

src: mkdirs
	make -C src

examples: mkdirs
	make -C examples


clean:
	rm bin -rf
	make -C examples clean
	make -C src clean
