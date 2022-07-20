FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Install some basic utilities
RUN apt-get update 
RUN apt-get update 
RUN apt-get update 
RUN apt-get install -y \
    git \
    zip \
    sudo \
    libx11-6 \
    build-essential \
    ca-certificates \
    wget \
    curl \
    tmux \
    htop \
    nano \
    vim

# ####################################################################################
# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Install Miniconda
WORKDIR /home/user
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/Miniconda3-py38_4.9.2-Linux-x86_64.sh\
 && ~/Miniconda3-py38_4.9.2-Linux-x86_64.sh -b -p ~/miniconda \
 && rm ~/Miniconda3-py38_4.9.2-Linux-x86_64.sh
ENV PATH=/home/user/miniconda/condabin/:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN conda init

# Create a Python 3.6 environment
RUN conda create -y --name py36 python=3.6.9 && conda clean -ya
RUN echo "conda activate py36" >> .bashrc
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN sudo apt-get install -y libsm6
RUN sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    libgl1-mesa-glx

RUN conda install -y cudatoolkit=10.0
RUN pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install pandas
RUN mkdir WORK

RUN pip install absl-py==0.9.0 albumentations==0.4.3 cachetools==4.0.0 certifi==2019.11.28 chardet==3.0.4 cycler==0.10.0 decorator==4.4.2 future==0.18.2 google-auth==1.11.3 google-auth-oauthlib==0.4.1 grpcio==1.27.2 idna==2.9 imageio==2.8.0 imgaug==0.2.6 joblib==0.14.1 kiwisolver==1.1.0 Markdown==3.2.1 matplotlib==3.0.3 networkx==2.4 numpy==1.15.4 oauthlib==3.1.0 opencv-python==4.2.0.32 opencv-python-headless==4.2.0.32 Pillow==5.3.0 protobuf==3.11.3 pyasn1==0.4.8 pyasn1-modules==0.2.8 pyparsing==2.4.6 python-dateutil==2.7.3 PyWavelets==1.1.1 PyYAML==5.3.1 requests==2.23.0 requests-oauthlib==1.3.0 rsa==4.0 scikit-image==0.15.0 scikit-learn==0.22.2.post1 scipy==1.1.0 six==1.14.0 sklearn==0.0 tensorboard==2.1.0 torch==1.1.0 torchvision==0.2.2.post3 tqdm==4.27.0 urllib3==1.25.8 Werkzeug==1.0.0
