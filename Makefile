TEST_MODEL := https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
TAG := $(shell git tag|sort|tail -1)
IMAGE := runpod_kobold

build:
	docker build -t $(IMAGE) .

test:
	docker run --rm -ti -p 5002:5002 \
	-e RUNPOD_PUBLIC_IP=1.2.3.4 \
	-e RUNPOD_TCP_PORT_22=2345 \
	-e MODEL=$(TEST_MODEL) \
	-e PUBLIC_KEY=123 \
	-e CTX=16384 \
	-e QUANTKV=1 \
	-e DRYRUN=1 \
	$(IMAGE)

push:
	docker tag $(IMAGE) ghcr.io/jojje/$(IMAGE):$(TAG)
	docker push ghcr.io/jojje/$(IMAGE):$(TAG)

