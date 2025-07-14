#!/usr/bin/env bash
set -euo pipefail

PROJECT=
DATASET=
REGION=                # adjust if needed
OUTFILE=

# Find all .sql files
shopt -s nullglob
sql_files=( *.sql )
if [[ ${#sql_files[@]} -eq 0 ]]; then
  echo "❌ No .sql files found in $(pwd)" >&2
  exit 1
fi


# Write header
cat > "$OUTFILE" <<EOF
query_file,job_id,state,elapsed_s,slot_s,cpu_s,bytes_processed_mb,bytes_billed_mb,shuffle_mb
EOF

for SQLFILE in "${sql_files[@]}"; do
  echo "▶️  Running $SQLFILE…"
  
  # Temporarily disable 'exit on error' so we can capture the job ID even if the query fails
  set +e
  bq --project_id="$PROJECT" \
     --dataset_id="$DATASET" \
     query \
       --use_legacy_sql=false \
       --nouse_cache \
     < "$SQLFILE" \
    > /dev/null
  QUERY_EXIT=$?
  set -e

  # Grab the very last job (whether SUCCESS or FAILURE)
  read JOB_ID JOB_STATE < <(
    bq ls -j -n 1 --location="$REGION" \
      | awk 'NR>2 { print $1, $3; exit }'
  )
  echo "   🎯 Job $JOB_ID → $JOB_STATE"
  
  if [[ "$JOB_STATE" != "SUCCESS" ]]; then
    # On failure, write N/A for metrics
    echo "$SQLFILE,$JOB_ID,$JOB_STATE,N/A,N/A,N/A,N/A,N/A,N/A" >> "$OUTFILE"
    continue
  fi

  # On success, fetch metrics
  RAW_CSV=$( bq --project_id="$PROJECT" \
                --location="$REGION" \
                query \
                  --use_legacy_sql=false \
                  --format=csv \
                <<EOF
SELECT
  job_id,
  state,
  TIMESTAMP_DIFF(end_time, start_time, SECOND)                   AS elapsed_s,
  total_slot_ms    / 1000.0                                     AS slot_s,
  total_bytes_processed  / (1024.0*1024.0)                      AS bytes_processed_mb,
  total_bytes_billed     / (1024.0*1024.0)                      AS bytes_billed_mb
FROM
  \`${PROJECT}.region-${REGION}.INFORMATION_SCHEMA.JOBS_BY_PROJECT\`
WHERE
  job_id = '$JOB_ID';
EOF
  )

  # 4) Strip header line
  METRICS_CSV=$(echo "$RAW_CSV" | tail -n +2)

  # 5) Append to results.csv
  echo "${SQLFILE},${METRICS_CSV}" >> "$OUTFILE"
done

echo "✅ All done! Results in $OUTFILE"

