-- TPC-DS Query 95, Redshift syntax

WITH ws_wh AS (
  SELECT
    ws1.ws_order_number,
    ws1.ws_warehouse_sk AS wh1,
    ws2.ws_warehouse_sk AS wh2
  FROM web_sales AS ws1
  JOIN web_sales AS ws2
    ON ws1.ws_order_number   = ws2.ws_order_number
   AND ws1.ws_warehouse_sk  <> ws2.ws_warehouse_sk
)
SELECT
  COUNT(DISTINCT ws1.ws_order_number)   AS order_count,
  SUM(ws1.ws_ext_ship_cost)             AS total_shipping_cost,
  SUM(ws1.ws_net_profit)                AS total_net_profit
FROM web_sales           AS ws1
JOIN date_dim            AS d    ON ws1.ws_ship_date_sk = d.d_date_sk
JOIN customer_address    AS ca   ON ws1.ws_ship_addr_sk = ca.ca_address_sk
JOIN web_site            AS wsit ON ws1.ws_web_site_sk  = wsit.web_site_sk
WHERE
  d.d_date BETWEEN DATE '2000-04-01'
               AND DATEADD(day, 60, DATE '2000-04-01')
  AND ca.ca_state           = 'IN'
  AND wsit.web_company_name = 'pri'
  AND ws1.ws_order_number IN (
    SELECT ws_order_number
    FROM ws_wh
  )
  AND ws1.ws_order_number IN (
    SELECT wr.wr_order_number
    FROM web_returns AS wr
    JOIN ws_wh      AS wh
      ON wr.wr_order_number = wh.ws_order_number
  )
ORDER BY
  order_count
LIMIT 100;
