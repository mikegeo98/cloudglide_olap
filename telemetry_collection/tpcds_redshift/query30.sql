-- TPC-DS Query 30, Redshift syntax

WITH customer_total_return AS (
  SELECT
    wr.wr_returning_customer_sk   AS ctr_customer_sk,
    ca.ca_state                   AS ctr_state,
    SUM(wr.wr_return_amt)         AS ctr_total_return
  FROM web_returns      AS wr
  JOIN date_dim         AS d  ON wr.wr_returned_date_sk  = d.d_date_sk
  JOIN customer_address AS ca ON wr.wr_returning_addr_sk = ca.ca_address_sk
  WHERE d.d_year = 2000
  GROUP BY
    wr.wr_returning_customer_sk,
    ca.ca_state
)
SELECT
  c.c_customer_id,
  c.c_salutation,
  c.c_first_name,
  c.c_last_name,
  c.c_preferred_cust_flag,
  c.c_birth_day,
  c.c_birth_month,
  c.c_birth_year,
  c.c_birth_country,
  c.c_login,
  c.c_email_address,
  dr.d_date                 AS c_last_review_date,
  ctr1.ctr_total_return
FROM customer_total_return AS ctr1
  -- bring in the customer record
  JOIN customer         AS c  ON ctr1.ctr_customer_sk = c.c_customer_sk

  -- filter to Indiana via the customer’s current address
  JOIN customer_address AS ca ON c.c_current_addr_sk  = ca.ca_address_sk
                             AND ca.ca_state         = 'IN'

  -- get the actual last_review_date from the date_dim
  JOIN date_dim         AS dr ON c.c_last_review_date_sk = dr.d_date_sk

WHERE ctr1.ctr_total_return > (
  -- per-state 120% threshold
  SELECT AVG(ctr2.ctr_total_return) * 1.2
  FROM customer_total_return AS ctr2
  WHERE ctr2.ctr_state = ctr1.ctr_state
)
ORDER BY
  c.c_customer_id,
  c.c_salutation,
  c.c_first_name,
  c.c_last_name,
  c.c_preferred_cust_flag,
  c.c_birth_day,
  c.c_birth_month,
  c.c_birth_year,
  c.c_birth_country,
  c.c_login,
  c.c_email_address,
  dr.d_date,
  ctr1.ctr_total_return
LIMIT 100;
