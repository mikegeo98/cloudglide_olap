#!/usr/bin/env bash
HOST=
PORT=
DB=
USER=
PGPASSWORD=
# export PGOPTIONS="-c search_path=imdb2025 -c enable_result_cache_for_session=off"
export PGOPTIONS="-c enable_result_cache_for_session=off"

# List all benchmark subdirectories
BENCHMARK_DIRS=(
  complex2
)

# Number of runs per directory
NUM_RUNS=5

for benchmark in "${BENCHMARK_DIRS[@]}"; do
  # Check that directory exists
  if [[ ! -d "$benchmark" ]]; then
    echo "Directory '$benchmark' not found, skipping…" >&2
    continue
  fi

  for run in $(seq 1 $NUM_RUNS); do
    LOGFILE="run_${benchmark}_${run}.log"
    CSVFILE="results_${benchmark}_${run}.csv"

    # Initialize (or overwrite) log file and CSV for this run
    echo "Benchmark '$benchmark', run #${run} started at $(date)" > "$LOGFILE"
    echo "query_file_name,runtime_s" > "$CSVFILE"

    echo "=== Starting benchmark '$benchmark' iteration #${run} ===" | tee -a "$LOGFILE"

    # Loop over all .sql files in the current benchmark directory
    for sql in "$benchmark"/*.sql; do
      # If there are no .sql files, skip
      [[ -e "$sql" ]] || { echo "No .sql files in '$benchmark', skipping…" | tee -a "$LOGFILE"; break; }

      echo "--------------------------------------------------" | tee -a "$LOGFILE"
      echo "Running $sql…"       | tee -a "$LOGFILE"

      start_time=$(date +%s.%N)

      if PGPASSWORD=$PGPASSWORD psql \
           --host="$HOST" \
           --port="$PORT" \
           --username="$USER" \
           --dbname="$DB" \
           --file="$sql" \
           --set ON_ERROR_STOP=on \
           &>> "$LOGFILE"; then
        status="SUCCESS"
      else
        status="FAILED"
      fi

      end_time=$(date +%s.%N)
      elapsed=$(echo "$end_time - $start_time" | bc)

      # Pull the query ID(s) that ran during this interval
      query_ids=$(PGPASSWORD=$PGPASSWORD psql \
           --host="$HOST" \
           --port="$PORT" \
           --username="$USER" \
           --dbname="$DB" \
           --tuples-only --quiet \
           --command="
      SELECT query
        FROM stl_query
       WHERE userid = (SELECT usesysid FROM pg_user WHERE usename = '$USER')
         AND starttime BETWEEN TIMESTAMP 'epoch' + $start_time * INTERVAL '1 second'
                           AND TIMESTAMP 'epoch' + $end_time   * INTERVAL '1 second'
       ORDER BY starttime DESC
       LIMIT 5;
      ")

      echo "Result:   $status"    | tee -a "$LOGFILE"
      echo "Elapsed:  ${elapsed}s" | tee -a "$LOGFILE"
      echo "Query ID: $query_ids"  | tee -a "$LOGFILE"

      # Append to this run's CSV file
      echo "$query_ids,$elapsed" >> "$CSVFILE"
    done

    echo "=== Finished benchmark '$benchmark' iteration #${run} at $(date) ===" | tee -a "$LOGFILE"
    echo ""  # blank line for readability in logs
  done
done

