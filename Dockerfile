FROM docker.io/library/ubuntu:20.04

LABEL maintainer="Marco Morais <marcoemorais@yahoo.com>"

# Base system including toolchain and dependencies.
RUN apt-get update && apt-get upgrade -y && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    ffmpeg \
    libgl1-mesa-dev \
    pandoc \
    python3 \
    python3-pip \
    xvfb

# Upgrade pip to support newer format wheels used by some packages.
RUN pip3 install --upgrade pip

# Copy requirements file into container image.
COPY requirements.txt /

RUN pip3 install -r requirements.txt

# By convention, all development in /src.
WORKDIR /src

# Expose port for jupyter.
ENV JUPYTER_PORT 8888
EXPOSE ${JUPYTER_PORT}

# PyVista settings for headless display.
ENV DISPLAY=:99.0
ENV PYVISTA_OFF_SCREEN=true
ENV PYVISTA_USE_PANEL=true

# Script used to start PyVista applications.
COPY start.sh /app/start.sh
