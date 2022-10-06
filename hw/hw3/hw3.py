import sys

import findspark
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from rolling_avg_transformer import RollingAverageTransform

findspark.init()
# line below is for use if findspark.init() function requires path to Spark.
# findspark.init('/Users/nia/Spark/spark-3.3.0-bin-hadoop3', edit_rc=True)


def main():
    appName = "PySpark-MariaDB baseball"
    master = "local"
    # Create Spark session
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

    database = "baseball"
    user = "root"
    password = ""  # pragma: allowlist secret
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    # SQL to join batter_counts and game tables
    sql = """
        select
            bc.batter,
            bc.game_id,
            bc.hit,
            bc.atBat,
            DATE(g.local_date) as local_date
        from batter_counts as bc
        join game as g
        on g.game_id = bc.game_id
        group by bc.batter, bc.game_id
    """

    # Create a data frame by reading data from MariaDB
    batter_game_df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .option("query", sql)
        .load()
    )

    batter_game_df.persist(StorageLevel.DISK_ONLY)

    rolling_avg_transform = RollingAverageTransform()
    pipeline = Pipeline(stages=[rolling_avg_transform])

    # fit batter_game_df to pipeline
    model = pipeline.fit(batter_game_df)
    battering_100days_rolling_df = model.transform(batter_game_df)
    battering_100days_rolling_df.show()
    return


if __name__ == "__main__":
    sys.exit(main())
