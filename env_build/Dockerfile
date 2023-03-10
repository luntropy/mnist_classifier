FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

MAINTAINER Betina Borisoova (b.borisowa17@gmail.com)

ENV color_prompt=yes

WORKDIR /app
COPY . .

RUN apt update || true
RUN apt install -y python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install poetry

RUN poetry install

CMD poetry run python app.py
