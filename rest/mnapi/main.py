from typing import Union
from typing import List, Dict
from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from pyspark.sql import SparkSession

from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.ml.linalg import VectorUDT

"""
TODO:
Endpoints
1. /accounts/ list of known mastodon users and ID numbers
2. /tf-idf/user-ids/<user-id> --returns vocabulary words, and values
3. /tf-idf/user-ids/<user-id>/neighbors
    -returns 10 closest users by cosine distance of vectorized TF-IDF.
"""


app = FastAPI()
# Load the Parquet file containing the TF-IDF values
# try:
#     tfidf_df = pd.read_parquet('tfidf.parquet')
# except:
#     print('There was an error reading the parquet')
#     tf_idf_df = None
import os
# Initialize the TF-IDF vectorizer and fit it to the corpus of text data
spark = SparkSession.builder \
    .appName("spark-worker")\
    .master("spark://spark-master:7077")\
    .config("spark.executor.instances", 1)\
    .config("spark.cores.max", 2)\
    .getOrCreate()

schema = StructType([
    StructField("id", StringType(), True),
    StructField("words", ArrayType(StringType()), True),
    StructField("features", VectorUDT(), True)
])
# tfidf = spark.read.parquet('/opt/warehouse/tf_idf3.parquet') 
tfidf = spark.read.schema(schema).parquet('/opt/warehouse/tf_idf3.parquet')
tfidf_df=tfidf.toPandas()
print(tfidf_df.columns)
print(tfidf_df.head())
# vectorizer = TfidfVectorizer()

# corpus = tfidf_df['words']
# tfidf_matrix = vectorizer.fit_transform(corpus)

# # Compute the cosine similarity matrix for all users
# cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


tf_idf_df = None
@app.get("/")
def read_root():
    print("in read_root")  
    # spark = SparkSession.builder \
    # .appName("spark-worker")\
    # .master("spark://spark-master:7077")\
    # .config("spark.executor.instances", 1)\
    # .config("spark.cores.max", 2)\
    # .getOrCreate()
    # tfidf = spark.read.parquet('/opt/warehouse/tf_idf3.parquet') 
    # vectorizer = TfidfVectorizer()
    # print(tfidf.columns)
    # print(tfidf.head())
    # tfidf_df = tfidf.toPandas()
    # print(cosine_sim_matrix)
    try:
        print("This is a try")
        print(os.listdir('..'))
        print(os.listdir('/opt/warehouse/'))
        print(os.listdir('../warehouse/'))
        # tfidf_df = tfidf.toPandas()
        print("The tdidf populated successfully")

        message = 'The import worked'

    except:
        print('There was an error reading the parquet')
        message = 'The import failed'
        tf_idf_df = None
    spark.stop()
    return {"Import status": message}


# Lists all of the known Mastodon accounts in the data-set.
@app.get('/mstdn-nlp/api/v1/accounts/')
def get_accounts() -> List[Dict[str, str]]:
    users_list = tfidf[['username', 'id']].to_dict('records')
    return users_list

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/sparktest")
def spark_test():
    from pyspark.sql import SparkSession
    spark_session = SparkSession.builder\
    .appName("rest-test")\
    .master("spark://spark-master:7077")\
    .config("spark.cores.max", 2)\
    .getOrCreate()
    try:
        wc2 = spark_session.read.parquet('/opt/warehouse/wordcounts.parquet/')
        wc2.createOrReplaceTempView("wordcounts")
        query = "SELECT * FROM wordcounts WHERE LEN(word) > 4 ORDER BY count DESC"
        ans = spark_session.sql(query).limit(10)
        list_of_dicts = ans.rdd.map(lambda row: row.asDict()).collect()
        return list_of_dicts
    finally:
        spark_session.stop()


# @app.get("/sparktest_tf_idf")
# def spark_test():
#     from pyspark.sql import SparkSession
#     spark = SparkSession.builder \
#         .appName("rest-test")\
#         .master("spark://spark-master:7077")\
#         .config("spark.executor.instances", 1)\
#         .config("spark.cores.max", 2)\
#         .getOrCreate()
#     try:
#         tf_idf = spark_session.read.parquet('/opt/warehouse/tf_idf.parquet/')
#         wc2.createOrReplaceTempView("wordcounts")
#         query = "SELECT * FROM wordcounts WHERE LEN(word) > 4 ORDER BY count DESC"
#         ans = spark_session.sql(query).limit(10)
#         list_of_dicts = ans.rdd.map(lambda row: row.asDict()).collect()
#         return list_of_dicts
#     finally:
#         spark_session.stop()