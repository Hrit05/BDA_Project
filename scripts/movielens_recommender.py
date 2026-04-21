from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# -------------------------------
# STEP 0: Setup
# -------------------------------
os.makedirs("output", exist_ok=True)
sns.set(style="whitegrid")

# -------------------------------
# STEP 1: Spark Session
# -------------------------------
spark = SparkSession.builder.appName("MovieLensRecommender").getOrCreate()

# -------------------------------
# STEP 2: Load Data
# -------------------------------
ratings = spark.read.csv(
    "data/ml-1m/ratings.dat",
    sep="::",
    inferSchema=True
).toDF("user_id", "movie_id", "rating", "timestamp")

movies = spark.read.csv(
    "data/ml-1m/movies.dat",
    sep="::",
    inferSchema=True
).toDF("movie_id", "title", "genre")

print("Data Loaded Successfully")

# -------------------------------
# STEP 3: EDA
# -------------------------------
print("Total Ratings:", ratings.count())
print("Unique Users:", ratings.select("user_id").distinct().count())
print("Unique Movies:", ratings.select("movie_id").distinct().count())

ratings_pd = ratings.limit(20000).toPandas()

# Ratings Distribution
plt.figure()
sns.histplot(ratings_pd["rating"], bins=5)
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig("output/rating_distribution.png")

# Top Movies
top_movies = ratings.groupBy("movie_id").count() \
    .orderBy("count", ascending=False).limit(10)

top_movies_pd = top_movies.join(movies, "movie_id").toPandas()

plt.figure()
sns.barplot(x="count", y="title", data=top_movies_pd)
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie")
plt.savefig("output/top_movies.png")

# User Activity
user_activity = ratings.groupBy("user_id").count()
user_pd = user_activity.limit(20000).toPandas()

plt.figure()
sns.histplot(user_pd["count"], bins=30)
plt.title("User Activity Distribution")
plt.xlabel("Ratings per User")
plt.ylabel("Frequency")
plt.savefig("output/user_activity.png")

# Average Rating Distribution
avg_rating = ratings.groupBy("movie_id").avg("rating")
avg_pd = avg_rating.limit(10000).toPandas()

plt.figure()
sns.histplot(avg_pd["avg(rating)"], bins=10)
plt.title("Average Movie Rating Distribution")
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.savefig("output/avg_movie_rating.png")

# -------------------------------
# STEP 4: Train-Test Split
# -------------------------------
train, test = ratings.randomSplit([0.8, 0.2])

# -------------------------------
# STEP 5: ALS Model
# -------------------------------
als = ALS(
    userCol="user_id",
    itemCol="movie_id",
    ratingCol="rating",
    coldStartStrategy="drop",
    rank=10,
    maxIter=10,
    regParam=0.1
)

model = als.fit(train)

# -------------------------------
# STEP 6: Evaluation
# -------------------------------
predictions = model.transform(test)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print("RMSE:", rmse)

# -------------------------------
# STEP 7: Recommendations (All Users)
# -------------------------------
user_recs = model.recommendForAllUsers(5)

print("Sample Recommendations:")
user_recs.show(5, False)

# -------------------------------
# STEP 8: Recommendations with Movie Names
# -------------------------------
rec_exploded = user_recs.select(
    col("user_id"),
    explode("recommendations").alias("rec")
)

rec_final = rec_exploded.select(
    col("user_id"),
    col("rec.movie_id"),
    col("rec.rating")
)

rec_named = rec_final.join(movies, "movie_id")

print("Recommendations with Movie Names:")
rec_named.show(10, False)

# -------------------------------
# STEP 9: Similar Movies (FIXED)
# -------------------------------

# Extract ALS item factors
item_factors = model.itemFactors
item_pd = item_factors.toPandas()

movies_pd = movies.toPandas()

# Cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Similar movies function
def get_similar_movies(movie_id, top_n=5):
    target_vec = item_pd[item_pd["id"] == movie_id]["features"].values

    if len(target_vec) == 0:
        return []

    target_vec = target_vec[0]

    similarities = []

    for _, row in item_pd.iterrows():
        sim = cosine_similarity(target_vec, row["features"])
        similarities.append((row["id"], sim))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    similarities = [x for x in similarities if x[0] != movie_id]

    return similarities[:top_n]

# Pick a valid movie ID from ALS
input_movie_id = item_pd["id"].iloc[0]

similar_movies = get_similar_movies(input_movie_id, 5)

print(f"\nMovies similar to Movie ID {input_movie_id}:\n")

for mid, score in similar_movies:
    movie_name = movies_pd[movies_pd["movie_id"] == mid]["title"].values
    
    if len(movie_name) > 0:
        print(f"{movie_name[0]} (Similarity: {score:.3f})")

# -------------------------------
# STEP 10: Single User Recommendation
# -------------------------------
single_user = ratings.select("user_id").limit(1)
user_specific = model.recommendForUserSubset(single_user, 5)

print("Single User Recommendation:")
user_specific.show(truncate=False)

# -------------------------------
# STEP 11: Save Output
# -------------------------------
rec_named.write.mode("overwrite").csv("output/final_recommendations", header=True)

print("Project Completed Successfully!")
