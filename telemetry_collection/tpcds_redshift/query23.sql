-- TPC-DS Query 23, Redshift syntax
WITH frequent_ss_items AS (
  SELECT
    substring(i.i_item_desc, 1, 30) AS itemdesc,
    i.i_item_sk                    AS item_sk,
    d.d_date                       AS solddate,
    COUNT(*)                       AS cnt
  FROM store_sales      ss
  JOIN date_dim         d  ON ss.ss_sold_date_sk = d.d_date_sk
  JOIN item             i  ON ss.ss_item_sk       = i.i_item_sk
  WHERE d.d_year IN (1998, 1999, 2000, 2001)
  GROUP BY
    substring(i.i_item_desc, 1, 30),
    i.i_item_sk,
    d.d_date
  HAVING COUNT(*) > 4
),
max_store_sales AS (
  SELECT MAX(csales) AS tpcds_cmax
  FROM (
    SELECT
      c.c_customer_sk,
      SUM(ss.ss_quantity * ss.ss_sales_price) AS csales
    FROM store_sales ss
    JOIN customer     c  ON ss.ss_customer_sk = c.c_customer_sk
    JOIN date_dim     d  ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE d.d_year IN (1998, 1999, 2000, 2001)
    GROUP BY c.c_customer_sk
  ) x
),
best_ss_customer AS (
  SELECT
    c.c_customer_sk,
    SUM(ss.ss_quantity * ss.ss_sales_price) AS ssales
  FROM store_sales ss
  JOIN customer     c  ON ss.ss_customer_sk = c.c_customer_sk
  GROUP BY c.c_customer_sk
  HAVING
    SUM(ss.ss_quantity * ss.ss_sales_price)
      > 0.95 * (SELECT tpcds_cmax FROM max_store_sales)
)
SELECT
  combined.c_last_name,
  combined.c_first_name,
  combined.sales
FROM (
  SELECT
    c2.c_last_name,
    c2.c_first_name,
    SUM(cs.cs_quantity * cs.cs_list_price) AS sales
  FROM catalog_sales cs
  JOIN customer       c2 ON cs.cs_bill_customer_sk = c2.c_customer_sk
  JOIN date_dim       d2 ON cs.cs_sold_date_sk     = d2.d_date_sk
  WHERE
    d2.d_year = 1998
    AND d2.d_moy = 6
    AND cs.cs_item_sk IN (SELECT item_sk FROM frequent_ss_items)
    AND cs.cs_bill_customer_sk IN (SELECT c_customer_sk FROM best_ss_customer)
  GROUP BY c2.c_last_name, c2.c_first_name

  UNION ALL

  SELECT
    c3.c_last_name,
    c3.c_first_name,
    SUM(ws.ws_quantity * ws.ws_list_price) AS sales
  FROM web_sales ws
  JOIN customer       c3 ON ws.ws_bill_customer_sk = c3.c_customer_sk
  JOIN date_dim       d3 ON ws.ws_sold_date_sk     = d3.d_date_sk
  WHERE
    d3.d_year = 1998
    AND d3.d_moy = 6
    AND ws.ws_item_sk IN (SELECT item_sk FROM frequent_ss_items)
    AND ws.ws_bill_customer_sk IN (SELECT c_customer_sk FROM best_ss_customer)
  GROUP BY c3.c_last_name, c3.c_first_name
) AS combined
ORDER BY combined.c_last_name,
         combined.c_first_name,
         combined.sales
LIMIT 100;
