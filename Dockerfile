FROM ubuntu:20.04

RUN apt-get update &&\
    apt-get install -y wget &&\
    apt-get install -y openbabel &&\
    apt-get -y install software-properties-common &&\
    apt-get -y install python3-pip

WORKDIR /funnel
COPY . .

RUN python3 -m pip install -r requirements.txt

EXPOSE 80
WORKDIR /funnel/app
CMD python3 -m uvicorn main:app --host 0.0.0.0 --port=80





