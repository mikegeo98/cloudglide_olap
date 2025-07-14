-- 5. UPDATE a subset of records
UPDATE test_orders
SET total_amount = total_amount * 1.10
WHERE order_date < CURRENT_DATE - 180;