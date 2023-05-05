FROM apache/spark-py:v3.4.0
USER root
RUN apt-get update && apt-get install -y python3
RUN pip install numpy pandas pyspark 
RUN pip3 install --upgrade setuptools
RUN pip3 install numpy pandas pyspark
WORKDIR /opt/app
COPY spark-worker.py ./spark_code_new/
# ENTRYPOINT [ "python" ]
CMD ["python3", "./spark_code_new/spark-worker.py"]

