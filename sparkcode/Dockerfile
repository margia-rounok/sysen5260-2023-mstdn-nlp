FROM apache/spark-py:v3.4.0
USER root
RUN pip install numpy==1.24.3
RUN pip install pandas==2.0.1
WORKDIR /opt/app
COPY spark_worker_2.py .

VOLUME /opt/warehouse
CMD ["python3","/opt/app/spark_worker_2.py"]
