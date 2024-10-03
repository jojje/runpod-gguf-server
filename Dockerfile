FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# add convenience commands
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    aria2 htop less nvtop vim-tiny \
 && rm -rf /var/lib/apt/lists/*

# add llamacpp for trouble shooting and auto-tuning
ENV LLAMACPP="b3870"
RUN cd /opt \
 && git clone https://github.com/ggerganov/llama.cpp \
 && cd llama.cpp \
 && git checkout -b ${LLAMACPP} ${LLAMACPP}
RUN mkdir /opt/cmake && cd /opt/cmake \
 && curl -Lo cmake.sh https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-linux-x86_64.sh \
 && bash cmake.sh --skip-license \
 && cd /opt/llama.cpp \
 && /opt/cmake/bin/cmake -B build -DBUILD_SHARED_LIBS=1 -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=all \
 && cd build \
 && make -j $(nproc) \
 && make install \
 && cd .. \
 && ldconfig /usr/local/lib \
 && rm -rf build /opt/cmake


# use a fast model downloader
RUN pip3 install "huggingface_hub[cli]" "huggingface_hub[hf_transfer]" hf-transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# use koboldcpp for model serving
ENV KOBOLD=1.75.2
RUN curl -Lo /usr/local/bin/koboldcpp \
    https://github.com/LostRuins/koboldcpp/releases/download/v${KOBOLD}/koboldcpp-linux-x64-cuda1210 \
 && chmod +x /usr/local/bin/koboldcpp

# hook into the runpod bootstrap chain
RUN cat > /post_start.sh <<'EOF'
#!/usr/bin/bash -e
exec /usr/bin/python3 -u /init.py
EOF
RUN chmod +x /post_start.sh

# add model downloading and bootstrap logic
RUN mkdir -p /models
COPY init.py /init.py
