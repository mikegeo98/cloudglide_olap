-- TPC-DS Q36, Redshift syntax (ORDER BY ordinals)
SELECT
  SUM(ss.ss_net_profit) * 1.0
    / SUM(ss.ss_ext_sales_price)      AS gross_margin,
  i.i_category,
  i.i_class,
  (GROUPING(i.i_category) + GROUPING(i.i_class)) AS lochierarchy,
  RANK() OVER (
    PARTITION BY
      (GROUPING(i.i_category) + GROUPING(i.i_class)),
      CASE WHEN GROUPING(i.i_class) = 0 THEN i.i_category END
    ORDER BY
      SUM(ss.ss_net_profit) * 1.0
        / SUM(ss.ss_ext_sales_price)
      ASC
  ) AS rank_within_parent
FROM store_sales AS ss
JOIN date_dim    AS d  ON ss.ss_sold_date_sk = d.d_date_sk
JOIN item        AS i  ON ss.ss_item_sk      = i.i_item_sk
JOIN store       AS s  ON ss.ss_store_sk     = s.s_store_sk
WHERE
  d.d_year = 2000
  AND s.s_state = 'TN'
GROUP BY ROLLUP(i.i_category, i.i_class)
ORDER BY
  4 DESC,                -- lochierarchy
  CASE WHEN 4 = 0        -- when lochierarchy = 0 (grand total), order by category
       THEN i.i_category 
  END,
  rank_within_parent
LIMIT 100;
