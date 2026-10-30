# ID2221 — Flight ETA Prediction Engine

Gradient Boosted Trees model that predicts remaining flight time and landing time for inbound aircraft (default hub: Frankfurt **EDDF**).

## Prerequisites

- **Java 17+** (required by PySpark)
- **Python 3.9+** (3.12 needs `setuptools` from `requirements.txt` for PySpark)
- **Docker** (only for the live Kafka / Cassandra pipeline)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS with Homebrew OpenJDK, point Spark at Java 17 if `java` is not already on your `PATH`:

```bash
export JAVA_HOME="$(/usr/libexec/java_home -v 17 2>/dev/null || echo /opt/homebrew/opt/openjdk@17)"
export PATH="$JAVA_HOME/bin:$PATH"
```

## Run (offline demo)

No Docker needed. From the repo root with the venv active:

```bash
make demo   # generate synthetic data → train → predict
```

This:

1. Writes synthetic history and mid-flight snapshots under `data/synthetic/`
2. Trains the GBT model to `models/synthetic/eta_gbt_pipeline`
3. Prints predicted landing times and `err_min` / MAE for the current-flight set

Useful individual steps:

| Command | What it does |
|---------|----------------|
| `make data` | Regenerate synthetic training + current flights |
| `make train` | Fit the model |
| `make predict` | Score `data/synthetic/current_flights` |

## Interact

### Streamlit dashboard (synthetic)

After `make demo`:

```bash
make viz
```

Open [http://localhost:8501](http://localhost:8501) for the inbound map, arrival board, and model vs naive ETA charts.

One-shot live OpenSky poll (no Kafka): `make viz-live`.

### Live pipeline (OpenSky → Kafka → Cassandra → UI)

Requires Docker. Bring the stack up once:

```bash
make pipeline   # synthetic seed → Docker (Kafka/HDFS/Spark/Cassandra) → train live model
```

Then in three terminals:

```bash
make producer        # OpenSky → Kafka every 60s
make stream-live     # Kafka → ETA every 60s → Cassandra
make viz-cassandra   # Streamlit reading live predictions
```

UI: filter by arrival airport and search by callsign / icao24.

Optional: query latest ETAs in Cassandra:

```bash
docker compose exec cassandra cqlsh -e \
  "SELECT icao24, callsign, pred_remaining_min, pred_landing_time FROM flights.eta_latest WHERE arr_airport='EDDF';"
```
