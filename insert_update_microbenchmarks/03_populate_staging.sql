-- 3. Populate staging with bulk data (e.g., 100k rows) on Redshift
INSERT INTO staging_orders
SELECT
  row_number() OVER ()      AS order_id,
  (random() * 10000)::BIGINT AS customer_id,
  CURRENT_DATE - (random() * 365)::INT     AS order_date,
  (random() * 1000)::NUMERIC(10,2)         AS total_amount
FROM stv_blocklist
LIMIT 10000000;
