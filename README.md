# ID2221-Project

## OpenSky Kafka producer

The producer polls OpenSky Network's `GET /states/all` endpoint and publishes one
JSON message per aircraft to Kafka. Messages use `icao24` as the Kafka key and
are sent to the `opensky-states` topic by default.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Kafka, create the topic if needed, and run the producer from the project
root:

```bash
python src/producer/producer.py
```

Configuration is available through environment variables:

```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_TOPIC=opensky-states \
POLL_INTERVAL_SECONDS=15 \
python src/producer/producer.py
```

Use `RUN_ONCE=true` to fetch and publish one batch, which is useful for testing.
OpenSky applies rate limits, so keep the polling interval conservative.

## Run Kafka locally with KRaft

The included [docker-compose.yml](docker-compose.yml) starts a single-node Kafka
4.0 cluster in KRaft mode. It has no ZooKeeper dependency and exposes Kafka on
`localhost:9092`.

Start Kafka:

```bash
docker compose up -d
```

Create the topic:

```bash
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh \
	--create \
	--topic opensky-states \
	--bootstrap-server localhost:9092 \
	--partitions 1 \
	--replication-factor 1
```

Start the producer in another terminal:

```bash
.venv/bin/python src/producer/producer.py
```

Watch the messages:

```bash
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
	--topic opensky-states \
	--from-beginning \
	--bootstrap-server localhost:9092
```

Stop Kafka while keeping its data:

```bash
docker compose down
```

To remove the stored topic data as well, use `docker compose down -v`.