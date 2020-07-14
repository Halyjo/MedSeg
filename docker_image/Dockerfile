FROM "pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime"
WORKDIR /tmp/
COPY requirements.txt .
RUN for i in $( cat requirements.txt ); do pip install $i; done
WORKDIR /
