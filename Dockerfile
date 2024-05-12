FROM ubuntu:latest
LABEL authors="roberttkach"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3-pip python3-venv libgl1-mesa-glx libsm6 libxext6

COPY . .

RUN python3 -m venv /app/venv
RUN /bin/bash -c "source /app/venv/bin/activate && pip install -r requirements.txt"

CMD ["/app/venv/bin/python3", "main.py"]
