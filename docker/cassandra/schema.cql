CREATE KEYSPACE IF NOT EXISTS flights
  WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

-- Run with: make cassandra
-- Arrivals board: newest prediction per aircraft for each arrival airport.
-- Rows expire after 1 h without a refresh (landed / out of range).
--   SELECT * FROM flights.eta_latest WHERE arr_airport = 'EDDF';
CREATE TABLE IF NOT EXISTS flights.eta_latest (
  arr_airport        text,
  icao24             text,
  callsign           text,
  ts                 bigint,
  dist_to_arr_km     double,
  altitude           double,
  velocity           double,
  vertical_rate      double,
  pred_remaining_min double,
  pred_landing_ts    bigint,
  pred_landing_time  timestamp,
  PRIMARY KEY ((arr_airport), icao24)
) WITH default_time_to_live = 3600;

-- Every prediction ever made for an aircraft (newest first), kept 7 days.
--   SELECT * FROM flights.eta_history WHERE icao24 = '3c65c1' LIMIT 20;
CREATE TABLE IF NOT EXISTS flights.eta_history (
  icao24             text,
  ts                 bigint,
  arr_airport        text,
  callsign           text,
  dist_to_arr_km     double,
  altitude           double,
  velocity           double,
  vertical_rate      double,
  pred_remaining_min double,
  pred_landing_ts    bigint,
  pred_landing_time  timestamp,
  PRIMARY KEY ((icao24), ts)
) WITH CLUSTERING ORDER BY (ts DESC) AND default_time_to_live = 604800;
