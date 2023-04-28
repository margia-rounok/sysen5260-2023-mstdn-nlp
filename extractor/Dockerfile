FROM python:3.8-slim-buster
USER root
RUN pip install requests
WORKDIR /opt/data
COPY extract.py .

CMD [ "python3", "extract.py" ]
