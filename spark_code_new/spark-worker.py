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
    print("about to load data")
    data = spark.read.json("/opt/data/*.json")

    print("loaded data correctly")

    # Use concat_ws() to combine the array of strings into a single column
    data = data.withColumn("content", concat_ws(" ", "content"))

    # Use groupBy() and concat_ws() to combine the strings for rows with the same ID
    data = data.groupBy("id").agg(concat_ws(" ", collect_list("content")).alias("combined_content"))

    # Tokenize content column
    tokenizer = Tokenizer(inputCol="combined_content", outputCol="words")
    data = tokenizer.transform(data)

    # Compute Term Frequencies
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    data = hashingTF.transform(data)

    # Compute Inverse Document Frequencies
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(data)
    data = idfModel.transform(data)

    # Convert sparse vectors to dense vectors
    to_dense = lambda v: DenseVector(v.toArray()) if isinstance(v, SparseVector) else v
    to_dense_udf = udf(to_dense, VectorUDT())
    data = data.withColumn("features", to_dense_udf("features"))

    # Write to file
    data.write.mode("overwrite").parquet(path="/opt/warehouse/tf_idf3.parquet",mode="overwrite")
    time.sleep(300)
# return

# import time

# if __name__ == '__main__':
#     print("in the main file")
#     while True:
#         print("in the while loop")
#         main()
#         time.sleep(300)