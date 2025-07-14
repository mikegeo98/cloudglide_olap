-- 6. DELETE old records
DELETE FROM test_orders
WHERE order_date < CURRENT_DATE - 365;