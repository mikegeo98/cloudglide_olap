WITH
-- 1) CPU time per query
cpu AS (
  SELECT
    query,
    SUM(cpu_time)::double precision / 1e6 AS total_cpu_sec
  FROM stl_query_metrics
  WHERE query > ID_START
    AND query < ID_END
  GROUP BY query
),

-- 2) “Other” stages (not scan/dist/bcast)
other_slice AS (
  SELECT
    query,
    step,
    DATEDIFF(milliseconds, start_time, end_time) AS dur_ms
  FROM svl_query_report
  WHERE query > ID_START
    AND query < ID_END
    AND label NOT ILIKE '%scan%'
    AND label NOT ILIKE 'dist%'
    AND label NOT ILIKE 'bcast%'
),
other_step AS (
  SELECT
    query,
    step,
    MAX(dur_ms) AS step_dur_ms
  FROM other_slice
  GROUP BY query, step
),
other AS (
  SELECT
    query,
    ROUND(SUM(step_dur_ms) / 1000.0, 3) AS other_wallclock_s
  FROM other_step
  GROUP BY query
),

-- 3) Scan stages
scan_slice AS (
  SELECT
    query,
    step,
    bytes,
    DATEDIFF(milliseconds, start_time, end_time) AS dur_ms
  FROM svl_query_report
  WHERE query > ID_START
    AND query < ID_END
    AND label ILIKE '%scan%'
),
scan_step AS (
  SELECT
    query,
    step,
    MAX(dur_ms) AS step_dur_ms
  FROM scan_slice
  GROUP BY query, step
),
scan_bytes AS (
  SELECT
    query,
    SUM(bytes) AS total_scan_bytes
  FROM scan_slice
  GROUP BY query
),
scan AS (
  SELECT
    sb.query,
    ROUND(sb.total_scan_bytes / 1024.0 / 1024.0, 3) AS total_scan_mb,
    ROUND(SUM(ss.step_dur_ms) / 1000.0, 3)          AS scan_wallclock_s
  FROM scan_bytes sb
  JOIN scan_step ss ON ss.query = sb.query
  GROUP BY sb.query, sb.total_scan_bytes
),

-- 4) Shuffle (dist + bcast) stages
dist_slice AS (
  SELECT
    query,
    step,
    bytes,
    DATEDIFF(milliseconds, start_time, end_time) AS dur_ms
  FROM svl_query_report
  WHERE query > ID_START
    AND query < ID_END
    AND (
      label ILIKE 'dist%' 
      OR label ILIKE 'bcast%'
    )
),
dist_step AS (
  SELECT
    query,
    step,
    MAX(dur_ms) AS step_dur_ms
  FROM dist_slice
  GROUP BY query, step
),
dist_bytes AS (
  SELECT
    query,
    SUM(bytes) AS total_shuffle_bytes
  FROM dist_slice
  GROUP BY query
),
shuffle AS (
  SELECT
    db.query,
    ROUND(db.total_shuffle_bytes / 1024.0 / 1024.0, 3) AS shuffle_mb,
    ROUND(SUM(ds.step_dur_ms) / 1000.0, 3)            AS shuffle_wallclock_s
  FROM dist_bytes db
  JOIN dist_step ds ON ds.query = db.query
  GROUP BY db.query, db.total_shuffle_bytes
),

-- 5) WLM execution time
exec_time AS (
  SELECT
    query,
    ROUND(total_exec_time / 1000.0 / 1000.0, 3) AS exec_wallclock_s
  FROM stl_wlm_query
  WHERE query > ID_START
    AND query < ID_END
)

-- final join
SELECT
  c.query,
  -- WLM execution time
  et.exec_wallclock_s,
  -- CPU
  c.total_cpu_sec,
  -- “Other” stages
  o.other_wallclock_s,
  -- Scan
  s.total_scan_mb,
  s.scan_wallclock_s,
  -- Shuffle
  sh.shuffle_mb,
  sh.shuffle_wallclock_s
FROM cpu      c
LEFT JOIN exec_time et ON et.query = c.query
LEFT JOIN other     o ON o.query  = c.query
LEFT JOIN scan      s ON s.query  = c.query
LEFT JOIN shuffle   sh ON sh.query = c.query
ORDER BY c.query;
