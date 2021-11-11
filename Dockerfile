FROM tensorflow/tensorflow:1.15.5-gpu-py3 AS build-tf

ADD . /work
WORKDIR /work
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip3 install torch
RUN git clone https://github.com/TachibanaYoshino/AnimeGANv2 && python convert_weights.py

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

COPY --from=build-tf /work /work
WORKDIR /work
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt