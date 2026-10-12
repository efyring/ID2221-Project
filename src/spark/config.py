"""Central Spark / Kafka runtime configuration (env-driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SparkConfig:
    """Resolved Spark settings used by session builders and streaming jobs."""

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
    retrain_every_n_batches: int
    ivy_path: str

    @property
    def kafka_package(self) -> str:
        return f"org.apache.spark:spark-sql-kafka-0-10_2.12:{self.version}"

    @classmethod
    def from_env(cls) -> SparkConfig:
        return cls(
            master=os.getenv("SPARK_MASTER", "spark://spark-master:7077"),
            version=os.getenv("SPARK_VERSION", "3.5.3"),
            driver_host=os.getenv("SPARK_DRIVER_HOST") or None,
            driver_bind_address=os.getenv("SPARK_DRIVER_BIND_ADDRESS", "0.0.0.0"),
            shuffle_partitions=os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "4"),
            log_level=os.getenv("SPARK_LOG_LEVEL", "WARN"),
            kafka_bootstrap_servers=os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"
            ),
            kafka_topic=os.getenv("KAFKA_TOPIC", "opensky-states"),
            train_data_path=os.getenv(
                "TRAIN_DATA_PATH", "./data/historical_opensky"
            ),
            model_output_path=os.getenv(
                "MODEL_OUTPUT_PATH", "./models/delay_gbt_pipeline"
            ),
            checkpoint_path=os.getenv(
                "CHECKPOINT_PATH", "./data/checkpoints/opensky_kafka"
            ),
            retrain_every_n_batches=int(os.getenv("RETRAIN_EVERY_N_BATCHES", "10")),
            ivy_path=os.getenv("IVY_HOME", "/opt/bitnami/spark/.ivy2"),
        )

    def session_configs(self, *, enable_kafka: bool = False) -> dict[str, str]:
        """SparkSession .config() key/value pairs for this environment."""
        configs: dict[str, str] = {
            "spark.sql.shuffle.partitions": self.shuffle_partitions,
            "spark.driver.bindAddress": self.driver_bind_address,
            # Absolute ivy path — Bitnami user 1001 often has no usable $HOME.
            "spark.jars.ivy": self.ivy_path,
        }
        if enable_kafka:
            configs["spark.jars.packages"] = self.kafka_package
        if self.driver_host:
            configs["spark.driver.host"] = self.driver_host
        return configs
