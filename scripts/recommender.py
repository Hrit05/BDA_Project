from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: Spark Session
# -------------------------------
spark = SparkSession.builder.appName("SpotifyRecommender").getOrCreate()

# -------------------------------
# STEP 2: Load Data
# -------------------------------
df = spark.read.csv("data/playlists.csv", header=True, inferSchema=True)

print("Original Data:")
df.show(5)

# -------------------------------
# STEP 3: Data Cleaning
# -------------------------------
df = df.select("Playlist", "Genre").dropna()

# -------------------------------
# STEP 4: EDA
# -------------------------------
print("Total Rows:", df.count())
print("Unique Playlists:", df.select("Playlist").distinct().count())
print("Unique Genres:", df.select("Genre").distinct().count())

# Genre distribution
genre_counts = df.groupBy("Genre").count().orderBy("count", ascending=False)
genre_pd = genre_counts.limit(10).toPandas()

plt.figure()
plt.bar(genre_pd["Genre"], genre_pd["count"])
plt.xticks(rotation=45)
plt.title("Top 10 Genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.savefig("output/genre_distribution.png")

# Playlist size distribution
playlist_counts = df.groupBy("Playlist").count()
playlist_pd = playlist_counts.limit(50).toPandas()

plt.figure()
plt.hist(playlist_pd["count"])
plt.title("Playlist Size Distribution")
plt.xlabel("Songs per Playlist")
plt.ylabel("Frequency")
plt.savefig("output/playlist_distribution.png")

# -------------------------------
# STEP 5: Prepare for ALS
# -------------------------------
df = df.withColumn("rating", lit(1))

# Convert categorical to numeric
user_indexer = StringIndexer(inputCol="Playlist", outputCol="user_id")
item_indexer = StringIndexer(inputCol="Genre", outputCol="item_id")

df = user_indexer.fit(df).transform(df)
df = item_indexer.fit(df).transform(df)

df = df.select("user_id", "item_id", "rating")

# -------------------------------
# STEP 6: Train-Test Split
# -------------------------------
train, test = df.randomSplit([0.8, 0.2])

# -------------------------------
# STEP 7: ALS Model
# -------------------------------
als = ALS(
    userCol="user_id",
    itemCol="item_id",
    ratingCol="rating",
    implicitPrefs=True,
    coldStartStrategy="drop",
    rank=10,
    maxIter=5,
    regParam=0.1
)

model = als.fit(train)

# -------------------------------
# STEP 8: Evaluation
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
# STEP 9: Recommendations
# -------------------------------
user_recs = model.recommendForAllUsers(5)

print("Sample Recommendations:")
user_recs.show(5, False)

# -------------------------------
# STEP 10: Save Output
# -------------------------------
user_recs.write.mode("overwrite").csv("output/recommendations", header=True)

print("Project Completed Successfully!")
