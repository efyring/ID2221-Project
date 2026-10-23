# Local demo
# Synthetic files live under data/synthetic and models/synthetic so they never mix with real data in data/historical_opensky.
export PYTHONPATH := src
export TRAIN_DATA_PATH ?= ./data/synthetic/historical_opensky
export AIRPORTS_PATH ?= ./data/synthetic/airports.csv
export MODEL_OUTPUT_PATH ?= ./models/synthetic/eta_gbt_pipeline
export PREDICT_INPUT_PATH ?= ./data/synthetic/current_flights

.PHONY: data train predict stream replay cassandra demo clean

data:      ## generate synthetic flights + airports under ./data/synthetic
	python src/data/generate_synthetic.py

train:     ## fit the model on TRAIN_DATA_PATH
	python src/ml/train.py

predict:   ## predict for PREDICT_INPUT_PATH and print the error
	python src/ml/predict.py

stream:    ## real-time: Kafka (localhost:9092) -> predictions every 15 s (CASSANDRA_HOST=localhost to also store them)
	KAFKA_BOOTSTRAP_SERVERS=$${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092} python src/ml/predict_stream.py

cassandra: ## start Cassandra, create keyspace + tables from docker/cassandra/schema.cql, verify
	docker compose up -d --wait cassandra
	docker compose exec -T cassandra cqlsh < docker/cassandra/schema.cql
	docker compose exec cassandra cqlsh -e "DESCRIBE TABLES" | grep -q eta_latest && echo "OK: keyspace flights is ready"

demo: data train predict

clean:
	rm -rf data/synthetic models/synthetic