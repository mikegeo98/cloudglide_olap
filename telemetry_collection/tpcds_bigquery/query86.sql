-- TPC-DS Query 86, Redshift syntax
SELECT
  SUM(ws.ws_net_paid)                                           AS total_sum,
  i.i_category,
  i.i_class,
  (GROUPING(i.i_category) + GROUPING(i.i_class))                AS lochierarchy,
  RANK() OVER (
    PARTITION BY
      (GROUPING(i.i_category) + GROUPING(i.i_class)),
      CASE WHEN GROUPING(i.i_class) = 0 THEN i.i_category END
    ORDER BY
      SUM(ws.ws_net_paid) DESC
  )                                                              AS rank_within_parent
FROM web_sales        AS ws
JOIN date_dim         AS d1 ON ws.ws_sold_date_sk = d1.d_date_sk
JOIN item             AS i  ON ws.ws_item_sk       = i.i_item_sk
WHERE
  d1.d_month_seq BETWEEN 1183 AND 1183 + 11
GROUP BY ROLLUP(i.i_category, i.i_class)
ORDER BY
  (GROUPING(i.i_category) + GROUPING(i.i_class)) DESC,
  CASE WHEN (GROUPING(i.i_category) + GROUPING(i.i_class)) = 0 THEN i.i_category END,
  rank_within_parent
LIMIT 100;
