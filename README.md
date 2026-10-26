# ID2221-Project

Kafka + Spark stack for OpenSky aircraft states, combined with a Gradient Boosted Trees (GBT) model that predicts remaining flight time and landing times at an airport (default: Frankfurt, **EDDF**; change via `ARRIVAL_AIRPORT`).

---

## Quick Start

### 1. Try End-to-End on Synthetic Data (No Docker Needed)
Requires Java 17/21 and Python:

```bash
pip install -r requirements.txt
make demo # Runs: make data, make train, make predict
```

* **`make data`**: Writes synthetic history (`data/synthetic/historical_opensky/`, ~3,000 flights, 500k snapshots) and current flight snapshots (`data/synthetic/current_flights/`, 200 unseen mid-flight snapshots).
* **`make train`**: Fits the GBT model on the synthetic folder, saves it to `models/eta_gbt_pipeline`, and logs test RMSE/MAE against a naive distance/speed baseline.
* **`make predict`**: Predicts landing times for `data/synthetic/current_flights`, printing `err_min` per flight and overall MAE.
* **`make stream`**: Real-time prediction from Kafka.
* **`make viz`**: Streamlit dashboard — inbound map, arrival board, model vs naive ETA, and error charts (uses synthetic predictions from `make demo`).

> **Note:** Synthetic data lives in `data/synthetic/` (wiped and rewritten on each `make data`), keeping it separate from real ingested data in `data/historical_opensky/`. Replace with real history once available.

### 2. Live Docker Stack
1. Start infrastructure and ingest stream:
   ```bash
   docker compose up -d --build
   ```
2. On the host, install producer dependencies and publish states:
   ```bash
   python -m pip install -r requirements.txt
   python src/producer/producer.py
   ```
3. **Monitor**:
   * Spark Master UI: [http://localhost:8080](http://localhost:8080)
   * Ingested Parquet data: `./data/historical_opensky/fetched_at_date=YYYY-MM-DD/`

---

## Architecture & Data Flow

| Component | Where it runs | Role |
|-----------|---------------|------|
| `kafka` / `kafka-init` | Docker | Message broker & topic initializer (`opensky-states`) |
| `spark-master` / `spark-worker` | Docker | Distributed compute cluster (`7077` / UI `8080`) |
| `ingest` | Docker | `src/spark/ingest.py` — Kafka → Parquet storage & periodic model training |
| `producer.py` | Host | Polls OpenSky HTTP API → Kafka |
| `predict_stream.py` | Host/Local | Real-time Kafka consumer → inbound filter → ETA predictions |

```text
LIVE: OpenSky API → producer (host) → Kafka → predict_stream.py → predicted landing times
TRAIN: historical flights (arrival airport + landing time) → train.py → model
```

---

## Data Requirements

* **Training Data**: Parquet at `TRAIN_DATA_PATH`, one row per position snapshot:
  * `icao24` (string): Aircraft ID
  * `arr_airport` (string): ICAO code of landing airport
  * `ts` (long): Snapshot time, Unix seconds UTC
  * `latitude`, `longitude` (double)
  * `baro_altitude`, `velocity`, `vertical_rate` (double; nulls filled with 0)
  * `actual_arrival_ts` (long): Actual landing timestamp
  * `on_ground` (boolean): Optional; grounded rows are dropped.
* **Airports CSV**: Located at `AIRPORTS_PATH` with header `icao,lat,lon`. (Departure airport is not needed).
* **Label & Features**: 
  * Label: `remaining_min = (actual_arrival_ts - ts) / 60`
  * Features: Distance to arrival airport, naive ETA (distance / speed), altitude, speed, vertical rate, UTC hour, weekday.

---

## Full live pipeline (OpenSky → Kafka → Spark → Cassandra → UI)

Requires Docker, Java 17+, and Python deps (`pip install -r requirements.txt`).

```bash
make pipeline          # train seed → Docker stack → Cassandra → HDFS dirs → train model
# then (keep running):
make producer          # Terminal 1: OpenSky HTTP → Kafka
make stream-live       # Terminal 2: Kafka → inbound EU hubs → GBT → parquet
make sync-cassandra    # push latest parquet → Cassandra (re-run periodically)
make viz-cassandra     # Terminal 3: Streamlit search UI
```

UI: filter by arrival airport (EDDF, EHAM, LFPG, …) and search callsign / icao24.  
Note: a given aircraft only appears while the inbound heuristic classifies it toward a hub.

## Streamlit Visualization (offline / synthetic)

```bash
make demo && make viz   # http://localhost:8501
```

Or one-shot live HTTP poll without Kafka: `make viz-live`.

---

## Real-Time Predictions & Cassandra Storage

### Running the Stream
Start Kafka and the producer, then run the streaming predictor:
```bash
python src/producer/producer.py # Terminal 1
make stream                     # Terminal 2: predictions every 15s
```

### Storing Predictions in Cassandra
```bash
make cassandra                   # Starts Cassandra, creates keyspace & tables
python src/producer/producer.py  # Terminal 1
CASSANDRA_HOST=localhost make stream # Terminal 2: predicts and writes every micro-batch
```
* **`flights.eta_latest`**: Newest prediction per aircraft (`arr_airport`, `icao24`), expiring after 1 h without refresh.
* **`flights.eta_history`**: Every prediction per aircraft (`icao24`, `ts DESC`), kept for 7 days.

Query latest arrivals via cqlsh:
```bash
docker compose exec cassandra cqlsh -e "SELECT icao24, callsign, pred_remaining_min, pred_landing_time FROM flights.eta_latest WHERE arr_airport='EDDF';"
```

---

## Configuration

| Variable | Cluster Default | Local Default | Description |
|----------|-----------------|---------------|-------------|
| `SPARK_MASTER` | `spark://spark-master:7077` | `local[*]` | Spark master URL |
| `TRAIN_DATA_PATH` | `hdfs://namenode:8020/data/historical_opensky` | `./data/historical_opensky` | Parquet historical data path |
| `AIRPORTS_PATH` | `hdfs://namenode:8020/data/airports.csv` | `./data/airports.csv` | Airports CSV file path |
| `MODEL_OUTPUT_PATH` | `hdfs://namenode:8020/models/eta_gbt_pipeline` | `./models/eta_gbt_pipeline` | Model output directory |
| `ARRIVAL_AIRPORT` | `EDDF` | `EDDF` | Target destination hub ICAO code |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | `localhost:9092` | Kafka bootstrap servers |
| `CASSANDRA_HOST` | `unset` (off) | `localhost` (to enable) | Cassandra host for sink |
| `DISPLAY_TZ` | `Europe/Berlin` | `Europe/Berlin` | Local wall-clock timezone for arrivals |

---

## Codebase Layout

* `src/producer/producer.py`: Polls OpenSky Network API → Kafka.
* `src/spark/{config,session,ingest,aggregate}.py`: Spark configuration, session, and streaming ingestion.
* `src/ml/features.py`: Schema, features, and label extraction.
* `src/ml/train.py`: Model training, evaluation, and saving.
* `src/ml/predict.py`: Batch inference and evaluation utilities.
* `src/ml/predict_stream.py`: Real-time Kafka consumer → inbound filter → prediction stream.
* `src/data/generate_synthetic.py`: Synthetic history and live data generator for testing.
* `src/viz/app.py`: Streamlit visualization of predicted remaining time / landing times.