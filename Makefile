.PHONY: build test all

build:
	cargo build-bpf

test:
	cargo test-sbf

test-print:
	cargo test-sbf -- --nocapture

test-debug:
	RUSTFLAGS="-C debug-assertions=on" cargo test-bpf -- --nocapture

all: build test

# Default target
default: all