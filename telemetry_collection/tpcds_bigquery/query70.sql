-- TPC-DS Q70, Redshift syntax
SELECT
  SUM(ss.ss_net_profit)                                                         AS total_sum,
  s.s_state,
  s.s_county,
  (GROUPING(s.s_state) + GROUPING(s.s_county))                                  AS lochierarchy,
  RANK() OVER (
    PARTITION BY (GROUPING(s.s_state) + GROUPING(s.s_county)),
                 CASE WHEN GROUPING(s.s_county) = 0 THEN s.s_state END
    ORDER BY SUM(ss.ss_net_profit) DESC
  )                                                                              AS rank_within_parent
FROM store_sales      AS ss
JOIN date_dim         AS d1 ON ss.ss_sold_date_sk = d1.d_date_sk
JOIN store            AS s  ON ss.ss_store_sk      = s.s_store_sk
WHERE
  d1.d_month_seq BETWEEN 1200 AND 1200 + 11
  AND s.s_state IN (
    SELECT s_state
    FROM (
      SELECT
        s2.s_state AS s_state,
        RANK() OVER (
          PARTITION BY s2.s_state
          ORDER BY SUM(ss2.ss_net_profit) DESC
        ) AS ranking
      FROM store_sales ss2
      JOIN store    s2  ON ss2.ss_store_sk      = s2.s_store_sk
      JOIN date_dim d2  ON ss2.ss_sold_date_sk  = d2.d_date_sk
      WHERE d2.d_month_seq BETWEEN 1200 AND 1200 + 11
      GROUP BY s2.s_state
    ) tmp1
    WHERE ranking <= 5
  )
GROUP BY ROLLUP(s.s_state, s.s_county)
ORDER BY
  (GROUPING(s.s_state) + GROUPING(s.s_county)) DESC,
  CASE WHEN (GROUPING(s.s_state) + GROUPING(s.s_county)) = 0 THEN s.s_state END,
  rank_within_parent
LIMIT 100;
