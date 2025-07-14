-- 1. Create main test table
CREATE TABLE test_orders (
  order_id       BIGINT PRIMARY KEY,
  customer_id    BIGINT,
  order_date     DATE,
  total_amount   DECIMAL(10,2)
);