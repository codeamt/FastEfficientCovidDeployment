FROM ubuntu:18.04

MAINTANER "annmargaret.tutu@icloud.com"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

COPY ./requirements.txt

RUN pip install -r requirements.txt

COPY ./src/ /

ENTRYPOINT [ "python" ]

CMD [ "start_server.py" ]