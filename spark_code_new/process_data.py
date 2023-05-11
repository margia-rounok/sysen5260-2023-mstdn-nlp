from pyspark.sql import SparkSession
from pyspark.sql.functions import log10, explode
import sys
import numpy

from pyspark.sql.functions import log10, concat_ws, flatten,collect_list

### Imports for TF-IDF 
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import SparseVector, VectorUDT, DenseVector
from pyspark.sql.functions import concat_ws, collect_list, udf

# ../bin/spark-submit  --master spark://spark-master:7077 countab.py
import time
# def main():
while True:
    # Initialize SparkSession
    print("about to create spark sessions")
    # spark = SparkSession.builder.appName("TF-IDF").getOrCreate()
    spark = SparkSession.builder \
        .appName("spark-worker")\
        .master("spark://spark-master:7077")\
        .config("spark.executor.instances", 1)\
        .config("spark.cores.max", 2)\
        .getOrCreate()

    # Load data from JSON file
    import os
    print("about to print directory")
    print(os.listdir())

    print("about to load data")
    data = spark.read.json("/opt/data/*.json")
    print("read file")
    data.show()
    print("loaded data correctly")

    # Use concat_ws() to combine the array of strings into a single column
    data = data.withColumn("content", concat_ws(" ", "content"))
    data.show()
    print("before groyp by")
    # Use groupBy() and concat_ws() to combine the strings for rows with the same ID
    data = data.groupBy("id").agg(concat_ws(" ", collect_list("content")).alias("combined_content"))

    # Tokenize content column
    tokenizer = Tokenizer(inputCol="combined_content", outputCol="words")
    data = tokenizer.transform(data)
    data.show()

    print("after tokenization")
    # Compute Term Frequencies
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    data = hashingTF.transform(data)
    data.show()

    print("after term frequencies")
    # Compute Inverse Document Frequencies
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(data)
    data = idfModel.transform(data)

    print("after idfModel")
    # Convert sparse vectors to dense vectors

    data.show()
    print("After convert to dense vector")
    data = data.drop("rawFeatures")
    data.show()
    print("After drop raw features")
    data = data.drop("combined_content")
    data.show()
    print("after drop combined content")
    to_dense = lambda v: DenseVector(v.toArray()) if isinstance(v, SparseVector) else v
    to_dense_udf = udf(to_dense, VectorUDT())
    data = data.withColumn("features", to_dense_udf("features"))
    data.show()
    print("after withColumn")
    data.printSchema()
    data.show(5)
    # Write to file
    data.write.parquet(path="/opt/warehouse/tf_idf3.parquet",mode="overwrite")
    #spark.stop()
    time.sleep(300)
# return

# import time

# if __name__ == '__main__':
#     print("in the main file")
#     while True:
#         print("in the while loop")
#         main()
#         time.sleep(300)