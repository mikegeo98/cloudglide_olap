-- TPC-DS Query 32, Redshift syntax
SELECT
  SUM(cs.cs_ext_discount_amt) AS excess_discount_amount
FROM catalog_sales   AS cs
JOIN item            AS i   ON cs.cs_item_sk      = i.i_item_sk
JOIN date_dim        AS d   ON cs.cs_sold_date_sk = d.d_date_sk
WHERE
  i.i_manufact_id = 610
  AND d.d_date BETWEEN DATE '2001-03-04'
                  AND DATEADD(day, 90, DATE '2001-03-04')
  AND cs.cs_ext_discount_amt > (
    SELECT 1.3 * AVG(cs2.cs_ext_discount_amt)
    FROM catalog_sales AS cs2
    JOIN date_dim      AS d2   ON cs2.cs_sold_date_sk = d2.d_date_sk
    WHERE
      cs2.cs_item_sk = cs.cs_item_sk
      AND d2.d_date BETWEEN DATE '2001-03-04'
                       AND DATEADD(day, 90, DATE '2001-03-04')
  )
LIMIT 100;
