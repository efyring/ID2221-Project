# Local demo + full live OpenSky → Kafka → Spark → Cassandra → Streamlit pipeline.
export PYTHONPATH := src
export TRAIN_DATA_PATH ?= ./data/synthetic/historical_opensky
export AIRPORTS_PATH ?= ./data/synthetic/airports.csv
export MODEL_OUTPUT_PATH ?= ./models/synthetic/eta_gbt_pipeline
export PREDICT_INPUT_PATH ?= ./data/synthetic/current_flights
export LIVE_STATES_PATH ?= ./data/live/current_states
export LIVE_PREDICTIONS_PATH ?= ./data/live/predictions.parquet
export VIZ_CACHE_PATH ?= ./data/synthetic/predictions.parquet

.PHONY: data train predict predict-live stream cassandra demo viz viz-live \
	stack-up seed-hdfs train-live producer stream-live pipeline viz-cassandra clean

data:      ## generate synthetic flights + airports under ./data/synthetic
	python src/data/generate_synthetic.py

train:     ## fit the model on TRAIN_DATA_PATH
	python src/ml/train.py

predict:   ## predict for PREDICT_INPUT_PATH and print the error
	python src/ml/predict.py

predict-live: ## one-shot OpenSky HTTP poll → inbound filter → predictions (no Kafka)
	AIRPORTS_PATH=./data/airports.csv \
	MODEL_OUTPUT_PATH=./models/synthetic/eta_gbt_pipeline \
	python src/ml/predict_opensky.py

stream:    ## Kafka → multi-hub ETA stream (set CASSANDRA_HOST=localhost to persist)
	KAFKA_BOOTSTRAP_SERVERS=$${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092} \
	AIRPORTS_PATH=./data/airports.csv \
	MODEL_OUTPUT_PATH=./models/synthetic/eta_gbt_pipeline \
	python src/ml/predict_stream.py

cassandra: ## start Cassandra + create flights.eta_latest / eta_history
	docker compose up -d --wait cassandra
	docker compose exec -T cassandra cqlsh -e "DROP TABLE IF EXISTS flights.eta_history; DROP TABLE IF EXISTS flights.eta_latest;" || true
	docker compose exec -T cassandra cqlsh < docker/cassandra/schema.cql
	docker compose exec cassandra cqlsh -e "DESCRIBE TABLES" | grep -q eta_latest && echo "OK: keyspace flights is ready"

demo: data train predict

stack-up: ## Kafka + HDFS + Spark + ingest (+ rebuild ingest image)
	docker compose up -d --build kafka kafka-init namenode datanode spark-master spark-worker ingest
	docker compose up -d --wait kafka namenode
	@echo "Stack up. Spark UI: http://localhost:8080"

seed-hdfs: ## upload airports + labelled training parquet to HDFS
	chmod +x scripts/seed_hdfs.sh
	./scripts/seed_hdfs.sh

train-live: ## train GBT on local labelled seed (synthetic or enriched), save under models/live + models/synthetic
	@test -d data/synthetic/historical_opensky || $(MAKE) data
	TRAIN_DATA_PATH=./data/synthetic/historical_opensky \
	AIRPORTS_PATH=./data/airports.csv \
	MODEL_OUTPUT_PATH=./models/live/eta_gbt_pipeline \
	SPARK_MASTER=local[*] \
	python src/ml/train.py
	@mkdir -p models/synthetic && rm -rf models/synthetic/eta_gbt_pipeline && cp -R models/live/eta_gbt_pipeline models/synthetic/eta_gbt_pipeline

producer: ## poll OpenSky → Kafka (foreground; Ctrl+C to stop)
	KAFKA_BOOTSTRAP_SERVERS=localhost:9092 python src/producer/producer.py

stream-live: ## Kafka → European hubs → model → local parquet (then: make sync-cassandra)
	KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
	AIRPORTS_PATH=./data/airports.csv \
	MODEL_OUTPUT_PATH=./models/synthetic/eta_gbt_pipeline \
	PREDICTIONS_PATH=./data/live/stream_predictions \
	STREAM_CHECKPOINT_PATH=./data/checkpoints/eta_stream \
	python src/ml/predict_stream.py

sync-cassandra: ## push newest stream parquet predictions into Cassandra
	CASSANDRA_HOST=localhost python src/viz/sync_cassandra.py

pipeline: data stack-up cassandra seed-hdfs train-live ## bring up full live path (then run producer + stream-live + viz-cassandra)
	@echo ""
	@echo "Pipeline ready. In three terminals run:"
	@echo "  make producer      # OpenSky → Kafka"
	@echo "  make stream-live   # Kafka → ETA → Cassandra"
	@echo "  make viz-cassandra # Streamlit search UI"
	@echo ""

viz:       ## Streamlit (parquet cache / synthetic)
	PYTHONPATH=src streamlit run src/viz/app.py --server.headless true

viz-live: predict-live
	VIZ_CACHE_PATH=$(LIVE_PREDICTIONS_PATH) \
	PREDICT_INPUT_PATH=$(LIVE_STATES_PATH) \
	AIRPORTS_PATH=./data/airports.csv \
	MODEL_OUTPUT_PATH=./models/synthetic/eta_gbt_pipeline \
	PYTHONPATH=src streamlit run src/viz/app.py --server.headless true

viz-cassandra: ## Streamlit reading live Cassandra eta_latest (European airport search)
	CASSANDRA_HOST=localhost \
	AIRPORTS_PATH=./data/airports.csv \
	VIZ_SOURCE=cassandra \
	PYTHONPATH=src streamlit run src/viz/app.py --server.headless true

clean:
	rm -rf data/synthetic models/synthetic data/live data/checkpoints/eta_stream
