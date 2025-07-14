-- 4. Bulk INSERT from staging into main table
INSERT INTO test_orders
SELECT * FROM staging_orders;