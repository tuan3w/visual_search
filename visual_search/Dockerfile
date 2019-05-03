FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get -yqq update && apt-get install -y --no-install-recommends \
      python \
      python-pip \
      python-dev \
      python-setuptools \
      python-opencv \
      python-tk;

RUN pip install --upgrade pip
RUN apt install -y --no-install-recommends curl

RUN mkdir /opt/visual_search
WORKDIR /opt/visual_search

COPY requirements.txt /opt/visual_search
RUN pip install -r requirements.txt

# Fix numpy install location issue (https://stackoverflow.com/questions/44888925/fatal-error-numpy-arrayobject-h-no-such-file-or-directory)
RUN ln -s /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy /usr/include/numpy

COPY . /opt/visual_search

# Setup Fast-RCNN libs
RUN cd lib && make

EXPOSE 5000

# Index data in elasticsearch and start visual search Flask app
CMD entrypoint.sh
