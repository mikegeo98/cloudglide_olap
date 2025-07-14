-- TPC-DS Query 99, Redshift syntax
SELECT
  SUBSTRING(w.w_warehouse_name, 1, 20)      AS warehouse_name_prefix,
  sm.sm_type                               AS ship_mode_type,
  cc.cc_name                               AS call_center_name,
  SUM(CASE WHEN DATEDIFF(day, d.d_date, d2.d_date) <= 30 THEN 1 ELSE 0 END)       AS days_30,
  SUM(CASE WHEN DATEDIFF(day, d.d_date, d2.d_date) BETWEEN 31 AND 60 THEN 1 ELSE 0 END)  AS days_31_60,
  SUM(CASE WHEN DATEDIFF(day, d.d_date, d2.d_date) BETWEEN 61 AND 90 THEN 1 ELSE 0 END)  AS days_61_90,
  SUM(CASE WHEN DATEDIFF(day, d.d_date, d2.d_date) BETWEEN 91 AND 120 THEN 1 ELSE 0 END) AS days_91_120,
  SUM(CASE WHEN DATEDIFF(day, d.d_date, d2.d_date) > 120 THEN 1 ELSE 0 END)      AS days_over_120
FROM catalog_sales     AS cs
JOIN date_dim          AS d   ON cs.cs_sold_date_sk = d.d_date_sk
JOIN date_dim          AS d2  ON cs.cs_ship_date_sk = d2.d_date_sk
JOIN warehouse         AS w   ON cs.cs_warehouse_sk = w.w_warehouse_sk
JOIN ship_mode         AS sm  ON cs.cs_ship_mode_sk = sm.sm_ship_mode_sk
JOIN call_center       AS cc  ON cs.cs_call_center_sk = cc.cc_call_center_sk
WHERE
  d.d_month_seq BETWEEN 1200 AND 1200 + 11
GROUP BY
  SUBSTRING(w.w_warehouse_name, 1, 20),
  sm.sm_type,
  cc.cc_name
ORDER BY
  warehouse_name_prefix,
  ship_mode_type,
  call_center_name
LIMIT 100;
