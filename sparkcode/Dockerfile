FROM apache/spark-py:v3.4.0
USER root
RUN pip install numpy pandas
WORKDIR /opt/app/
COPY spark-worker.py .
ENTRYPOINT [ "spark-submit" ]
CMD ["spark-submit","/opt/app/spark-worker.py"]
