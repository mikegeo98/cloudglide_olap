-- TPC-DS Query 50, Redshift syntax
SELECT
  s.s_store_name,
  s.s_company_id,
  s.s_street_number,
  s.s_street_name,
  s.s_street_type,
  s.s_suite_number,
  s.s_city,
  s.s_county,
  s.s_state,
  s.s_zip,
  SUM(CASE WHEN DATEDIFF(day, d1.d_date, d2.d_date) <= 30 THEN 1 ELSE 0 END)     AS days_30,
  SUM(CASE WHEN DATEDIFF(day, d1.d_date, d2.d_date) BETWEEN 31 AND 60 THEN 1 ELSE 0 END)  AS days_31_60,
  SUM(CASE WHEN DATEDIFF(day, d1.d_date, d2.d_date) BETWEEN 61 AND 90 THEN 1 ELSE 0 END)  AS days_61_90,
  SUM(CASE WHEN DATEDIFF(day, d1.d_date, d2.d_date) BETWEEN 91 AND 120 THEN 1 ELSE 0 END) AS days_91_120,
  SUM(CASE WHEN DATEDIFF(day, d1.d_date, d2.d_date) > 120 THEN 1 ELSE 0 END)     AS days_over_120
FROM store_sales    AS ss
JOIN store_returns  AS sr
  ON ss.ss_ticket_number = sr.sr_ticket_number
 AND ss.ss_item_sk      = sr.sr_item_sk
 AND ss.ss_customer_sk  = sr.sr_customer_sk
JOIN store          AS s
  ON ss.ss_store_sk = s.s_store_sk
JOIN date_dim       AS d1
  ON ss.ss_sold_date_sk     = d1.d_date_sk
JOIN date_dim       AS d2
  ON sr.sr_returned_date_sk = d2.d_date_sk
WHERE
  d2.d_year = 2002
  AND d2.d_moy  = 9
GROUP BY
  s.s_store_name,
  s.s_company_id,
  s.s_street_number,
  s.s_street_name,
  s.s_street_type,
  s.s_suite_number,
  s.s_city,
  s.s_county,
  s.s_state,
  s.s_zip
ORDER BY
  s.s_store_name,
  s.s_company_id,
  s.s_street_number,
  s.s_street_name,
  s.s_street_type,
  s.s_suite_number,
  s.s_city,
  s.s_county,
  s.s_state,
  s.s_zip
LIMIT 100;
