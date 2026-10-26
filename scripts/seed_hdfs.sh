#!/usr/bin/env bash
# Seed HDFS with airports + labelled training parquet for the Spark cluster.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NN="${NAMENODE_CONTAINER:-id2221-namenode}"

echo "Creating HDFS dirs…"
docker exec "$NN" hdfs dfs -mkdir -p /data /models /checkpoints /data/historical_opensky /data/train_labelled

echo "Uploading airports.csv…"
docker cp "$ROOT/data/airports.csv" "$NN":/tmp/airports.csv
docker exec "$NN" hdfs dfs -put -f /tmp/airports.csv /data/airports.csv

TRAIN_SRC="${TRAIN_SEED:-$ROOT/data/synthetic/historical_opensky}"
PART="$(find "$TRAIN_SRC" -name '*.parquet' | head -1 || true)"
if [[ -z "$PART" ]]; then
  echo "No training parquet under $TRAIN_SRC — run: make data"
  exit 1
fi
echo "Uploading labelled training parquet $PART → /data/train_labelled/"
docker cp "$PART" "$NN":/tmp/train_labelled.parquet
docker exec "$NN" hdfs dfs -put -f /tmp/train_labelled.parquet /data/train_labelled/

echo "HDFS contents:"
docker exec "$NN" hdfs dfs -ls -R /data | head -40
echo "OK"
