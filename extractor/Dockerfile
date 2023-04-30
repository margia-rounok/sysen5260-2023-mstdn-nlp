FROM python:3.10
USER root
RUN pip install requests
RUN mkdir /opt/data
WORKDIR /opt/app
COPY extract.py .
VOLUME /opt/data
CMD [ "python3", "/opt/app/extract.py" ]
