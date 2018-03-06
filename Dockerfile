FROM nvidia/cuda:9.1-devel

ADD $PWD/requirements.txt /requirements.txt
ADD $PWD/cudnn/libcudnn7_7.0.5.15-1+cuda9.1_amd64.deb /libcudnn7_7.0.5.15-1+cuda9.1_amd64.deb
ADD $PWD/cudnn/libcudnn7-dev_7.0.5.15-1+cuda9.1_amd64.deb /libcudnn7-dev_7.0.5.15-1+cuda9.1_amd64.deb

RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get install git -y
RUN apt-get install cmake -y
RUN apt-get install python3 -y 
RUN apt-get install python3-pip -y
RUN apt-get install libopenblas-dev liblapack-dev -y

RUN dpkg -i libcudnn7_7.0.5.15-1+cuda9.1_amd64.deb
RUN dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.1_amd64.deb

RUN pip3 install --upgrade pip
RUN pip3 install -U -r /requirements.txt

RUN git clone https://github.com/davisking/dlib
WORKDIR dlib
RUN python3 setup.py install

RUN pip3 install git+https://github.com/ageitgey/face_recognition_models

WORKDIR /app

ADD $PWD/*.py /app

EXPOSE 3001

CMD ["python3", "twins-server.py"]