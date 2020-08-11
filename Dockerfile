FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update -y && \
	apt-get install -y build-essential unzip wget libglib2.0-0 && \
	apt install python3.8 python3-pip -y


RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

COPY deploy/ /app/deploy/
WORKDIR /app

RUN ln -s /usr/bin/python3.8 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

ENV PATH="/root/.local/bin/:${PATH}"

RUN bash deploy/install_dependencies.sh



COPY . /app/

# RUN python setup.py install

ENTRYPOINT ["/bin/bash"]