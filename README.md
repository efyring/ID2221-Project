# ID2221-Project

Kafka + Spark continuous-learning stack for OpenSky aircraft states: a host-side
producer publishes to Kafka; a Spark Structured Streaming job appends Parquet and
periodically retrains a GBT delay model (proxy label).

## Quick start (compose-first)

1. Start Kafka, Spark master/worker, topic init, and the ingest job:

```bash
docker compose up -d --build
```

2. On the host, install producer deps and publish OpenSky states to `localhost:9092`:

```bash
python -m pip install -r requirements.txt
python src/producer/producer.py
```

3. Observe:

- Spark master UI: [http://localhost:8080](http://localhost:8080)
- Growing training data: `./data/historical_opensky/fetched_at_date=YYYY-MM-DD/`
- Checkpoints: `./data/checkpoints/opensky_kafka`
- Retrained model (every 10 micro-batches by default): `./models/delay_gbt_pipeline`

Stop while keeping Kafka broker volume data:

```bash
docker compose down
```

Remove the Kafka volume as well:

```bash
docker compose down -v
```

## Architecture

| Component | Where it runs | Role |
|-----------|---------------|------|
| `kafka` | Docker | Dual listeners: host `localhost:9092`, in-compose `kafka:29092` |
| `kafka-init` | Docker (one-shot) | Creates `opensky-states` if missing |
| `spark-master` / `spark-worker` | Docker | Spark cluster (`7077` / UI `8080`) |
| `ingest` | Docker | `src/spark/ingest.py` — Kafka → features → Parquet → periodic retrain |
| `producer.py` | Host | Polls OpenSky HTTP API → Kafka |

```text
OpenSky API → producer (host) → Kafka → ingest (Spark) → ./data + ./models
```

## Producer

Polls OpenSky Network `GET /states/all` and publishes one JSON message per
aircraft (`icao24` key) to `opensky-states`.

```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_TOPIC=opensky-states \
POLL_INTERVAL_SECONDS=15 \
python src/producer/producer.py
```

Use `RUN_ONCE=true` for a single fetch. Keep the poll interval conservative;
OpenSky rate-limits anonymous clients.

## Spark package (`src/spark`)

Centralized Spark runtime:

| Module | Role |
|--------|------|
| `spark/config.py` | Env-driven `SparkConfig` (master, Kafka, paths, shuffle, driver host) |
| `spark/session.py` | `create_spark_session(...)` shared by ingest and batch train |
| `spark/aggregate.py` | Per-aircraft `groupBy` aggregations for GBT training |
| `spark/ingest.py` | Structured Streaming job |

ML stays under `src/ml` (`features.py`, `train.py`).

## Ingest + continuous learning

`src/spark/ingest.py` uses `spark-sql-kafka`, parses with `OPENSKY_SCHEMA`, runs
`compute_features` + `add_proxy_label` (column `actual_delay_min` — kinematics
proxy, not schedule ground truth), appends each micro-batch to Parquet partitioned
by calendar date `fetched_at_date=YYYY-MM-DD` (from `fetched_at`, UTC), and every
`RETRAIN_EVERY_N_BATCHES` (default `10`) aggregates snapshots per `icao24` via
`spark.aggregate.aggregate_for_training` then calls `fit_and_save`.


Per-aircraft Spark aggregations (mean/min/max/stddev of distance, altitude,
velocity, vertical rate, descent gradient, plus sample count) are the GBT
feature inputs; the label is the mean proxy delay over the track.

| Variable | In-compose default | Host / offline |
|----------|--------------------|----------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | `localhost:9092` |
| `KAFKA_TOPIC` | `opensky-states` | same |
| `SPARK_MASTER` | `spark://spark-master:7077` | `local[*]` |
| `SPARK_DRIVER_HOST` | `ingest` | (unused for `local[*]`) |
| `TRAIN_DATA_PATH` | `/data/historical_opensky` | `./data/historical_opensky` |
| `MODEL_OUTPUT_PATH` | `/models/delay_gbt_pipeline` | `./models/delay_gbt_pipeline` |
| `CHECKPOINT_PATH` | `/data/checkpoints/opensky_kafka` | `./data/checkpoints/opensky_kafka` |
| `RETRAIN_EVERY_N_BATCHES` | `10` | same |

## Offline batch train

Against accumulated Parquet (paths above), from the Spark master container:

```bash
docker compose exec spark-master \
  python3 /opt/project/src/ml/train.py
```

Or locally with PySpark:

```bash
PYTHONPATH=src \
SPARK_MASTER=local[*] \
TRAIN_DATA_PATH=./data/historical_opensky \
MODEL_OUTPUT_PATH=./models/delay_gbt_pipeline \
python src/ml/train.py
```


## Manual topic create (optional)

`kafka-init` normally creates the topic. To create it by hand:

```bash
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh \
	--create \
	--if-not-exists \
	--topic opensky-states \
	--bootstrap-server localhost:9092 \
	--partitions 1 \
	--replication-factor 1
```

## Console consumer

```bash
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
	--topic opensky-states \
	--from-beginning \
	--bootstrap-server localhost:9092
```
