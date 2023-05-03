FROM apache/spark-py:v3.4.0
USER root
RUN pip install numpy==1.24.3
RUN pip install pandas==2.0.1
COPY spark-worker.py .
VOLUME /opt/warehouse
# CMD ["python3","/opt/app/spark_worker_2.py"]
CMD ["./sbin/start-worker.sh","spark://spark-master:7077"]
# CMD ["/bin/spark-submit", "--master", "spark://spark-master:7077", "work-dir/spark-worker.py"]