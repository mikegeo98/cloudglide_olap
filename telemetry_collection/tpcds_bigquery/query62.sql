-- TPC-DS Query 62, Redshift syntax
SELECT
  SUBSTRING(w.w_warehouse_name, 1, 20)      AS warehouse_name_prefix,
  sm.sm_type                               AS ship_mode_type,
  wsit.web_name                            AS web_site_name,
  SUM(CASE WHEN DATEDIFF(day, d_sold.d_date, d_ship.d_date) <= 30 THEN 1 ELSE 0 END)      AS days_30,
  SUM(CASE WHEN DATEDIFF(day, d_sold.d_date, d_ship.d_date) BETWEEN 31 AND 60 THEN 1 ELSE 0 END) AS days_31_60,
  SUM(CASE WHEN DATEDIFF(day, d_sold.d_date, d_ship.d_date) BETWEEN 61 AND 90 THEN 1 ELSE 0 END) AS days_61_90,
  SUM(CASE WHEN DATEDIFF(day, d_sold.d_date, d_ship.d_date) BETWEEN 91 AND 120 THEN 1 ELSE 0 END) AS days_91_120,
  SUM(CASE WHEN DATEDIFF(day, d_sold.d_date, d_ship.d_date) > 120 THEN 1 ELSE 0 END)     AS days_over_120
FROM web_sales          AS ws
JOIN date_dim           AS d_sold  ON ws.ws_sold_date_sk    = d_sold.d_date_sk
JOIN date_dim           AS d_ship  ON ws.ws_ship_date_sk    = d_ship.d_date_sk
JOIN warehouse          AS w       ON ws.ws_warehouse_sk    = w.w_warehouse_sk
JOIN ship_mode          AS sm      ON ws.ws_ship_mode_sk    = sm.sm_ship_mode_sk
JOIN web_site           AS wsit    ON ws.ws_web_site_sk     = wsit.web_site_sk
WHERE
  d_sold.d_month_seq BETWEEN 1222 AND 1222 + 11
GROUP BY
  SUBSTRING(w.w_warehouse_name, 1, 20),
  sm.sm_type,
  wsit.web_name
ORDER BY
  warehouse_name_prefix,
  ship_mode_type,
  web_site_name
LIMIT 100;
