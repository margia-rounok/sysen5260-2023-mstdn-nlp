from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.sql.functions import lower, regexp_replace, split
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("TF-IDF").getOrCreate()

# Read the CSV file from your data-lake
df = spark.read.json("path/to/csv/file.csv", header=True)

# Preprocess the text data
df = df.withColumn("toot_text", lower(regexp_replace("toot_text", "[^a-zA-Z\\s]", ""))) \
       .withColumn("toot_text", split("toot_text", "\\s+")) \
       .withColumn("toot_text", StopWordsRemover().setInputCol("toot_text").setOutputCol("toot_text_filtered").transform(df).drop("toot_text")) \

# Perform a term frequency count on the preprocessed text data for each user
cv = CountVectorizer(inputCol="toot_text_filtered", outputCol="tf", vocabSize=10000, minDF=2.0)
cvModel = cv.fit(df)
df_tf = cvModel.transform(df).select("user_id", "tf")

# Compute the inverse document frequency (IDF) for each term in the vocabulary
idf = IDF(inputCol="tf", outputCol="tf_idf")
idfModel = idf.fit(df_tf)
df_tfidf = idfModel.transform(df_tf).select("user_id", "tf_idf")

# Store the TF-IDF matrix in your warehouse volume as a Parquet file
df_tfidf.write.parquet("path/to/parquet/file.parquet")

# Stop the SparkSession
spark.stop()