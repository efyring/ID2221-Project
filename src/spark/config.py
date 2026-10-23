"""Central Spark / Kafka runtime configuration (env-driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SparkConfig:
    """Resolved Spark settings used by session builders and jobs."""

    master: str
    version: str
    driver_host: str | None
    driver_bind_address: str
    shuffle_partitions: str
    log_level: str
    kafka_bootstrap_servers: str
    kafka_topic: str
    train_data_path: str
    model_output_path: str
    checkpoint_path: str
    airports_path: str
    arrival_airport: str
    default_fs: str
    cassandra_host: str | None
    cassandra_port: str
    cassandra_keyspace: str
    ivy_path: str

    @property
    def kafka_package(self) -> str:
        return f"org.apache.spark:spark-sql-kafka-0-10_2.12:{self.version}"

    @property
    def cassandra_package(self) -> str:
        return "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1"

    @classmethod
    def from_env(cls) -> SparkConfig:
        return cls(
            master=os.getenv("SPARK_MASTER", "spark://spark-master:7077"),
            version=os.getenv("SPARK_VERSION", "3.5.3"),
            driver_host=os.getenv("SPARK_DRIVER_HOST") or None,
            driver_bind_address=os.getenv("SPARK_DRIVER_BIND_ADDRESS", "0.0.0.0"),
            shuffle_partitions=os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "4"),
            log_level=os.getenv("SPARK_LOG_LEVEL", "WARN"),
            kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
            kafka_topic=os.getenv("KAFKA_TOPIC", "opensky-states"),
            train_data_path=os.getenv(
                "TRAIN_DATA_PATH", "hdfs://namenode:8020/data/historical_opensky"
            ),
            model_output_path=os.getenv(
                "MODEL_OUTPUT_PATH", "hdfs://namenode:8020/models/eta_gbt_pipeline"
            ),
            checkpoint_path=os.getenv(
                "CHECKPOINT_PATH", "hdfs://namenode:8020/checkpoints/opensky_kafka"
            ),
            airports_path=os.getenv(
                "AIRPORTS_PATH", "hdfs://namenode:8020/data/airports.csv"
            ),
            arrival_airport=os.getenv("ARRIVAL_AIRPORT", "EDDF"),
            default_fs=os.getenv("DEFAULT_FS", "hdfs://namenode:8020"),
            cassandra_host=os.getenv("CASSANDRA_HOST") or None,  # unset -> Cassandra sink off
            cassandra_port=os.getenv("CASSANDRA_PORT", "9042"),
            cassandra_keyspace=os.getenv("CASSANDRA_KEYSPACE", "flights"),
            ivy_path=os.getenv("IVY_HOME", "/opt/bitnami/spark/.ivy2"),
        )

    def session_configs(
        self, *, enable_kafka: bool = False, enable_cassandra: bool = False
    ) -> dict[str, str]:
        """SparkSession .config() key/value pairs for this environment."""
        configs: dict[str, str] = {
            "spark.sql.shuffle.partitions": self.shuffle_partitions,
            "spark.driver.bindAddress": self.driver_bind_address,
            "spark.hadoop.fs.defaultFS": self.default_fs,
            # Hour/day-of-week features are computed in UTC.
            "spark.sql.session.timeZone": "UTC",
            # Absolute ivy path — Bitnami user 1001 often has no usable $HOME.
            "spark.jars.ivy": self.ivy_path,
        }
        packages = []
        if enable_kafka:
            packages.append(self.kafka_package)
        if enable_cassandra:
            packages.append(self.cassandra_package)
            configs["spark.cassandra.connection.host"] = self.cassandra_host or "localhost"
            configs["spark.cassandra.connection.port"] = self.cassandra_port
        if packages:
            configs["spark.jars.packages"] = ",".join(packages)
        if self.driver_host:
            configs["spark.driver.host"] = self.driver_host
        return configs