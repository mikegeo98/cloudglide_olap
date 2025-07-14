# Calibration of I/O Bandwidth Parameters

This guide explains how to empirically derive `DRAM-BW`, `SSD-BW` and `Net-BW` parameters for CloudGlide’s I/O model (Eq.~2) using only SQL on Amazon Redshift (or any compatible engine).

## Why These Steps?

To accurately model I/O service times, CloudGlide needs:

- **Cold-scan bandwidth** (`Net-BW`): end-to-end throughput when data is fetched from remote object storage (S3) through Spectrum/REDSHIFT → SSD → DRAM.  
- **Warm-scan bandwidth** (`SSD-BW` and/or `DRAM-BW`): throughput when data is already resident on local storage tiers.

By timing successive scans under different storage paths, we isolate each component.

---

## 1. Prepare a ~2 M-row table (~2 GiB total)

```sql
DROP TABLE IF EXISTS calib.big_data;
CREATE TABLE calib.big_data AS
WITH RECURSIVE nums(n) AS (
    SELECT 1
  UNION ALL
    SELECT n + 1 FROM nums WHERE n < 2097152  -- 2 097 152 rows
)
SELECT
  REPEAT(MD5(RANDOM()::varchar), 32) AS payload  -- ≈1 KiB per row
FROM nums;

ANALYZE calib.big_data;
```

---

## 2. Measure Local-Cache Bandwidth (SSD/DRAM)

Disable result caching and run two back-to-back scans:

```sql
SET enable_result_cache_for_session TO off;

-- Scan #1: populate SSD + DRAM cache
SELECT COUNT(*) FROM calib.big_data;  -- record duration as T_warmup

-- Scan #2: measure warm-cache throughput
SELECT COUNT(*) FROM calib.big_data;  -- record duration as T_warm
```

Compute:
```
S = 2 GiB
SSD-BW (or DRAM-BW) ≈ S / T_warm  (bytes/sec)
```
---

## 3. Measure Network Bandwidth (Net-BW)

Unload the same table to S3 (serial mode) and then read it back via Spectrum:

```sql
-- 3.1 UNLOAD to S3 (single slice, CSV):
UNLOAD ('SELECT payload FROM calib.big_data')
TO 's3://your-bucket/cloudglide/net-test/'
IAM_ROLE 'arn:aws:iam::590183820521:role/redshift'
ALLOWOVERWRITE
PARALLEL OFF
FORMAT AS CSV;

-- 3.2 Define an external schema and table:
CREATE EXTERNAL SCHEMA IF NOT EXISTS spectrum_calib
  FROM DATA CATALOG
    DATABASE 'spectrum_db'
    IAM_ROLE 'arn:aws:iam::590183820521:role/redshift';

CREATE EXTERNAL TABLE IF NOT EXISTS spectrum_calib.big_data (
  payload varchar(1024)
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://your-bucket/cloudglide/net-test/';
```

Now time two scans over S3:

```sql
SET enable_result_cache_for_session TO off;

-- Scan #1: cold S3 → SSD → DRAM path
SELECT COUNT(*) FROM spectrum_calib.big_data;  -- record duration as T_cold

-- Scan #2: now under local file cache (warm)
SELECT COUNT(*) FROM spectrum_calib.big_data;  -- record duration as T_s3_warm
```

Compute:
```
Net-BW ≈ S / T_cold           (bytes/sec)
Residual S3-cache BW ≈ S / T_s3_warm
```

## 4. Plug into CloudGlide

Edit your `config.json` to include the measured bandwidths:

```json
{
  "DRAM-BW": <measured DRAM-BW in bytes/sec>,
  "SSD-BW":  <measured SSD-BW in bytes/sec>,
  "Net-BW":  <measured Net-BW in bytes/sec>
}
```
