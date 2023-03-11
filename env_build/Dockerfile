FROM ubuntu:20.04

MAINTAINER Betina Borisoova (b.borisowa17@gmail.com)

ENV DEBIAN_FRONTEND="noninteractive"
ENV color_prompt=yes

WORKDIR /app
COPY . .

RUN apt update || true
RUN apt install -y python3-pip python3-opencv

RUN pip3 install --upgrade pip
RUN pip3 install poetry==1.4.0

RUN poetry config virtualenvs.create false 
RUN poetry install

CMD poetry run python3 app.py
