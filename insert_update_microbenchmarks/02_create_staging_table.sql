-- 2. Create staging table
CREATE TABLE staging_orders (
  order_id       BIGINT,
  customer_id    BIGINT,
  order_date     DATE,
  total_amount   DECIMAL(10,2)
);
