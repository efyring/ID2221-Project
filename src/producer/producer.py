"""Poll OpenSky Network and publish aircraft states to Kafka."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from typing import Any

import requests
from kafka import KafkaProducer


LOGGER = logging.getLogger(__name__)
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
STATE_FIELDS = (
	"icao24",
	"callsign",
	"origin_country",
	"time_position",
	"last_contact",
	"longitude",
	"latitude",
	"baro_altitude",
	"on_ground",
	"velocity",
	"true_track",
	"vertical_rate",
	"sensors",
	"geo_altitude",
	"squawk",
	"spi",
	"position_source",
	"category",
)


def _state_to_dict(state: list[Any], fetched_at: int) -> dict[str, Any]:
	"""Convert OpenSky's positional state vector into a self-describing record."""
	record = dict(zip(STATE_FIELDS, state, strict=False))
	record["fetched_at"] = fetched_at
	return record


def fetch_states(
	session: requests.Session,
	url: str = OPENSKY_STATES_URL,
	timeout: float = 30.0,
) -> tuple[int, Iterator[dict[str, Any]]]:
	"""Fetch the current aircraft states and return their API timestamp and records."""
	response = session.get(url, timeout=timeout)
	response.raise_for_status()
	payload = response.json()
	fetched_at = payload.get("time", int(time.time()))
	states = payload.get("states") or []
	return fetched_at, (_state_to_dict(state, fetched_at) for state in states)


def create_producer(bootstrap_servers: str) -> KafkaProducer:
	"""Create a Kafka producer that publishes JSON values and string keys."""
	return KafkaProducer(
		bootstrap_servers=bootstrap_servers.split(","),
		key_serializer=lambda key: key.encode("utf-8"),
		value_serializer=lambda value: json.dumps(value, separators=(",", ":")).encode(
			"utf-8"
		),
		acks="all",
	)


def publish_states(
	producer: KafkaProducer,
	topic: str,
	states: Iterator[dict[str, Any]],
) -> int:
	"""Publish records keyed by aircraft identifier and return the record count."""
	count = 0
	for state in states:
		icao24 = state.get("icao24")
		if not icao24:
			continue
		producer.send(topic, key=icao24, value=state)
		count += 1
	producer.flush()
	return count


def run() -> None:
	"""Poll OpenSky continuously, or once when RUN_ONCE is enabled."""
	logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
	topic = os.getenv("KAFKA_TOPIC", "opensky-states")
	bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
	interval = float(os.getenv("POLL_INTERVAL_SECONDS", "15"))
	run_once = os.getenv("RUN_ONCE", "false").lower() in {"1", "true", "yes"}
	session = requests.Session()
	producer = create_producer(bootstrap_servers)

	try:
		while True:
			try:
				_, states = fetch_states(session)
				count = publish_states(producer, topic, states)
				LOGGER.info("Published %d aircraft states to %s", count, topic)
			except requests.RequestException:
				LOGGER.exception("OpenSky request failed")
			except Exception:
				LOGGER.exception("Kafka publish failed")

			if run_once:
				break
			time.sleep(interval)
	finally:
		producer.close()


if __name__ == "__main__":
	run()
