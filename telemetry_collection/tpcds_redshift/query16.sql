-- TPC-DS Q16, Redshift syntax
SELECT
  COUNT(DISTINCT cs1.cs_order_number)      AS order_count,
  SUM(cs1.cs_ext_ship_cost)                AS total_shipping_cost,
  SUM(cs1.cs_net_profit)                   AS total_net_profit
FROM catalog_sales       AS cs1
JOIN date_dim            AS d
  ON cs1.cs_ship_date_sk = d.d_date_sk
JOIN customer_address    AS ca
  ON cs1.cs_ship_addr_sk = ca.ca_address_sk
JOIN call_center         AS cc
  ON cs1.cs_call_center_sk = cc.cc_call_center_sk
WHERE 
  d.d_date BETWEEN DATE '2002-03-01'
                AND DATEADD(day, 60, DATE '2002-03-01')
  AND ca.ca_state         = 'IA'
  AND cc.cc_county        = 'Williamson County'
  AND EXISTS (
    SELECT 1
    FROM catalog_sales AS cs2
    WHERE cs2.cs_order_number   = cs1.cs_order_number
      AND cs2.cs_warehouse_sk  <> cs1.cs_warehouse_sk
  )
  AND NOT EXISTS (
    SELECT 1
    FROM catalog_returns AS cr
    WHERE cr.cr_order_number    = cs1.cs_order_number
  )
ORDER BY order_count
LIMIT 100;
