-- TPC-DS Query 26, Redshift syntax
SELECT
  i.i_item_id                     AS item_id,
  AVG(cs.cs_quantity)             AS avg_quantity,
  AVG(cs.cs_list_price)           AS avg_list_price,
  AVG(cs.cs_coupon_amt)           AS avg_coupon_amt,
  AVG(cs.cs_sales_price)          AS avg_sales_price
FROM catalog_sales           AS cs
JOIN customer_demographics  AS cd
  ON cs.cs_bill_cdemo_sk     = cd.cd_demo_sk
JOIN date_dim               AS d
  ON cs.cs_sold_date_sk      = d.d_date_sk
JOIN item                   AS i
  ON cs.cs_item_sk           = i.i_item_sk
JOIN promotion              AS p
  ON cs.cs_promo_sk          = p.p_promo_sk
WHERE
  cd.cd_gender            = 'F'
  AND cd.cd_marital_status = 'W'
  AND cd.cd_education_status = 'Secondary'
  AND (p.p_channel_email = 'N' OR p.p_channel_event = 'N')
  AND d.d_year            = 2000
GROUP BY
  i.i_item_id
ORDER BY
  i.i_item_id
LIMIT 100;
