#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <BQ_JOB_ID>" >&2
  exit 1
fi

JOB_ID="$1"

# 1) Fetch the job metadata in JSON
#    --format=prettyjson ensures we get a JSON object, not a plain string.
META_JSON=$(bq --format=prettyjson show -j "$JOB_ID")

# 2) Use jq to pull out your five metrics
echo "$META_JSON" | jq -r '
  # compute elapsed as endTime - startTime
  def elapsed_ms: (.statistics.endTime | tonumber) - (.statistics.startTime | tonumber);

  [
    ("Elapsed (ms)",    elapsed_ms),
    ("Slot MS",         (.statistics.totalSlotMs // 0)),
    ("CPU MS",          (
                         (.statistics.totalCpuTimeValue // .statistics.totalCpuTime // "0")
                         | tonumber
                       )),
    ("Bytes Processed", (.statistics.totalBytesProcessed // 0)),
    ("Shuffle Bytes",   (.statistics.query.shuffleOutputBytes // 0))
  ]
  | .[]
  # print each pair as "Label: value"
  | "\(. [0]): \(. [1])"
'

