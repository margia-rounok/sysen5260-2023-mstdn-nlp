FROM python:3.8-slim-buster
USER root
RUN pip install requests
WORKDIR /opt/app/data
COPY extract.py extract.py

CMD [ "python", "extract.py" ]
