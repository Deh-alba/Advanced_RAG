FROM python:3.10.13


WORKDIR /app

COPY . /app

RUN apt-get update -y

#RUN apt install libreoffice -y

RUN pip install -r requirements.txt
#RUN pip install openai --upgrade


RUN useradd -ms /bin/bash andre && echo "root:root" | chpasswd


CMD [ "streamlit","run","app.py","--server.port","8080" ]