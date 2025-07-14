-- TPC-DS Query 92, Redshift syntax
SELECT
  SUM(ws.ws_ext_discount_amt) AS excess_discount_amount
FROM web_sales   AS ws
JOIN item        AS i   ON ws.ws_item_sk       = i.i_item_sk
JOIN date_dim    AS d   ON ws.ws_sold_date_sk  = d.d_date_sk
WHERE
  i.i_manufact_id = 718
  AND d.d_date BETWEEN DATE '2002-03-29'
                  AND DATEADD(day, 90, DATE '2002-03-29')
  AND ws.ws_ext_discount_amt > (
    SELECT 1.3 * AVG(ws2.ws_ext_discount_amt)
    FROM web_sales AS ws2
    JOIN date_dim  AS d2   ON ws2.ws_sold_date_sk = d2.d_date_sk
    WHERE
      ws2.ws_item_sk = i.i_item_sk
      AND d2.d_date BETWEEN DATE '2002-03-29'
                       AND DATEADD(day, 90, DATE '2002-03-29')
  )
ORDER BY
  SUM(ws.ws_ext_discount_amt)
LIMIT 100;
