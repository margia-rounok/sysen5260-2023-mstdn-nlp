FROM apache/spark-py:v3.4.0
USER root
RUN pip install numpy pandas
WORKDIR /opt/app/
COPY process_data.py .
ENTRYPOINT [ "spark-submit" ]
CMD ["spark-submit","/opt/app/process_data.py"]

