FROM apache/spark-py:v3.4.0
USER root
RUN pip install fastapi uvicorn[standard] scikit-learn pandas numpy pyarrow fastparquet pyspark
WORKDIR /opt/app
COPY server.sh .
RUN chmod a+rx server.sh
ADD mnapi/ mnapi/
ENTRYPOINT ["uvicorn", "mnapi.main:app", "--host", "0.0.0.0", "--port", "8000"]


