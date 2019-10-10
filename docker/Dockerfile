FROM ubuntu:18.04

ARG DEFAULT_WORKDIR=/tvm

RUN apt-get update && apt-get install -y --no-install-recommends \
	software-properties-common \
	git \
	python3 \
	python3-dev \
	python3-setuptools \
	python3-pip \
	gcc \
	libtinfo-dev \ 
	zlib1g-dev \
	build-essential \
	cmake \
	wget \
	tar \
	libxml2-dev

WORKDIR /opt

RUN wget http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
	tar xvf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
	rm clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz

WORKDIR /
RUN git clone --recursive https://github.com/dmlc/tvm

WORKDIR $DEFAULT_WORKDIR
RUN mkdir build
COPY docker.config.cmake build/config.cmake

WORKDIR $DEFAULT_WORKDIR/build
RUN cmake .. && make -j4


FROM ubuntu:18.04

ENV PYTHON_VERSION=3.6.8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PATH /opt/anaconda3/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	build-essential \
	software-properties-common \
    ca-certificates \
	wget \
    curl \
    htop \
    zip \
    unzip

RUN cd /opt && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash ./miniconda.sh -b -p /opt/anaconda3 && \
    rm miniconda.sh && \
    /opt/anaconda3/bin/conda clean -tipsy && \
    ln -s /opt/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
    
RUN conda config --set always_yes yes --set changeps1 no

RUN conda install pytorch-cpu torchvision-cpu -c pytorch
RUN pip install future pillow onnx opencv-python-headless tensorflow==1.14.0 onnxruntime
RUN pip install Cython && pip install pycocotools
RUN cd /tmp && \
    git clone https://github.com/mlperf/inference && \
    cd inference/loadgen && \
    pip install pybind11 && \
    CFLAGS="-std=c++14" python setup.py install && \
    rm -rf mlperf


FROM ubuntu:18.04

COPY --from=0 /tvm /opt/tvm
COPY --from=1 /tmp/inference /opt/mlperf-inference

ENV PYTHONPATH $DEFAULT_WORKDIR/python:$DEFAULT_WORKDIR/topi/python:$DEFAULT_WORKDIR/nnvm/python:${PYTHONPATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	python3 \
	python3-dev \
	python3-pip

# install TVM requirement
RUN python3 -m pip install numpy decorator attrs
WORKDIR /opt
RUN git clone https://github.com/uwsampl/relay-bench.git
