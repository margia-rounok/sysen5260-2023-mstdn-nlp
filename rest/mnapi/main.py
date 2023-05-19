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
schema = StructType([
    StructField("account", StringType(), True),
    StructField("combined_content", StringType(), True),
    StructField("words", ArrayType(StringType()), True),
    StructField("rawFeatures", VectorUDT(), True),
    StructField("features", VectorUDT(), True)
])
import os
# Initialize the TF-IDF vectorizer and fit it to the corpus of text data
# spark = SparkSession.builder \
#     .appName("rest-test")\
#     .master("spark://spark-master:7077")\
#     .config("spark.executor.instances", 1)\
#     .config("spark.cores.max", 2)\
#     .getOrCreate()


# # tfidf = spark.read.parquet('/opt/warehouse/tf_idf3.parquet') 
# tfidf = spark.read.schema(schema).parquet('/opt/warehouse/tf_idf3.parquet')
# tfidf_df=tfidf.toPandas()
# print(tfidf_df.columns)
# print(tfidf_df.head())
# vectorizer = TfidfVectorizer()

# corpus = tfidf_df['words']
# tfidf_matrix = vectorizer.fit_transform(corpus)

# # Compute the cosine similarity matrix for all users
# cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


tf_idf_df = None
@app.get("/")
def read_root():
    return {"Hello": "World"}

# 1 End point 
# Lists all of the known Mastodon accounts in the data-set.
@app.get('/mstdn-nlp/api/v1/accounts/')
def get_accounts() -> List[Dict[str, str]]:
    spark = SparkSession.builder \
    .appName("rest-test")\
    .master("spark://spark-master:7077")\
    .config("spark.executor.instances", 1)\
    .config("spark.cores.max", 2)\
    .getOrCreate()

    tfidf = spark.read.schema(schema).parquet('/opt/warehouse/tf_idf3.parquet')
    tfidf_df=tfidf.toPandas()
    users_list = tfidf_df[['account']].to_dict('records')
    return users_list

# 2 End point
@app.get("/tf-idf/user-ids/{user_id}")
def get_tf_idf(user_id: str) -> Dict[str, float]:
    """
    Returns the TF-IDF vector for the given user ID.
    """
    spark = SparkSession.builder \
    .appName("rest-test")\
    .master("spark://spark-master:7077")\
    .config("spark.executor.instances", 1)\
    .config("spark.cores.max", 2)\
    .getOrCreate()


    # tfidf = spark.read.parquet('/opt/warehouse/tf_idf3.parquet') 
    tfidf = spark.read.schema(schema).parquet('/opt/warehouse/tf_idf3.parquet')
    tfidf_df=tfidf.toPandas()
    print(tfidf_df.columns)
    print(tfidf_df.head())
    vectorizer = TfidfVectorizer()

    tf_idf_vector = tfidf_df[tfidf_df['account'] == user_id]['features'].values[0]
    tf_idf_vector = tf_idf_vector.toArray()
    tf_idf_dict = dict(zip(vectorizer.get_feature_names(), tf_idf_vector))
    return tf_idf_dict

# 3 End point
# Returns the 10 closest users by cosine distance of vectorized TF-IDF.
@app.get("/tf-idf/user-ids/{user_id}/neighbors")
def get_tf_idf_neighbors(user_id: str) -> List[Dict[str, float]]:
    """
    Returns the 10 closest users by cosine distance of vectorized TF-IDF.
    """
    spark = SparkSession.builder \
    .appName("rest-test")\
    .master("spark://spark-master:7077")\
    .config("spark.executor.instances", 1)\
    .config("spark.cores.max", 2)\
    .getOrCreate()


    # tfidf = spark.read.parquet('/opt/warehouse/tf_idf3.parquet') 
    tfidf = spark.read.schema(schema).parquet('/opt/warehouse/tf_idf3.parquet')
    tfidf_df=tfidf.toPandas()
    print(tfidf_df.columns)
    print(tfidf_df.head())
    vectorizer = TfidfVectorizer()

    corpus = tfidf_df['words']
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Compute the cosine similarity matrix for all users
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Get the index of the user ID in the TF-IDF matrix
    user_idx = tfidf_df[tfidf_df['account'] == user_id].index[0]

    # Get the cosine similarity scores of all users to the given user ID
    cosine_sim_scores = list(enumerate(cosine_sim_matrix[user_idx]))

    # Sort the users by their cosine similarity score
    sorted_cosine_sim_scores = sorted(cosine_sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 users
    top_10_users = sorted_cosine_sim_scores[1:11]

    # Get the user IDs of the top 10 users
    top_10_users_ids = [tfidf_df.iloc[i[0]]['account'] for i in top_10_users]

    # Get the TF-IDF vectors for the top 10 users
    top_10_users_tfidf = [tfidf_df.iloc[i[0]]['features'] for i in top_10_users]

    # Convert the TF-IDF vectors to lists
    top_10_users_tfidf = [i.toArray().tolist() for i in top_10_users_tfidf]

    # Zip the user IDs and TF-IDF vectors together
    top_10_users_tfidf = list(zip(top_10_users_ids, top_10_users_tfidf))

    # Convert the TF-IDF vectors to dictionaries
    top_10_users_tfidf = [dict(zip(vectorizer.get_feature_names(), i[1])) for i in top_10_users_tfidf]

    return top_10_users_tfidf



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

