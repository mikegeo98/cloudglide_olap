-- TPC-DS Q83, Redshift syntax
WITH sr_items AS (
  SELECT
    i.i_item_id                           AS item_id,
    SUM(sr.sr_return_quantity)            AS sr_item_qty
  FROM store_returns       AS sr
  JOIN item                AS i  ON sr.sr_item_sk       = i.i_item_sk
  JOIN date_dim            AS d  ON sr.sr_returned_date_sk = d.d_date_sk
  WHERE d.d_date IN (
    SELECT d2.d_date
    FROM date_dim AS d2
    WHERE d2.d_week_seq IN (
      SELECT d3.d_week_seq
      FROM date_dim AS d3
      WHERE d3.d_date IN (
        DATE '1999-06-30',
        DATE '1999-08-28',
        DATE '1999-11-18'
      )
    )
  )
  GROUP BY i.i_item_id
),
cr_items AS (
  SELECT
    i.i_item_id                           AS item_id,
    SUM(cr.cr_return_quantity)            AS cr_item_qty
  FROM catalog_returns    AS cr
  JOIN item                AS i  ON cr.cr_item_sk       = i.i_item_sk
  JOIN date_dim            AS d  ON cr.cr_returned_date_sk = d.d_date_sk
  WHERE d.d_date IN (
    SELECT d2.d_date
    FROM date_dim AS d2
    WHERE d2.d_week_seq IN (
      SELECT d3.d_week_seq
      FROM date_dim AS d3
      WHERE d3.d_date IN (
        DATE '1999-06-30',
        DATE '1999-08-28',
        DATE '1999-11-18'
      )
    )
  )
  GROUP BY i.i_item_id
),
wr_items AS (
  SELECT
    i.i_item_id                           AS item_id,
    SUM(wr.wr_return_quantity)            AS wr_item_qty
  FROM web_returns        AS wr
  JOIN item                AS i  ON wr.wr_item_sk       = i.i_item_sk
  JOIN date_dim            AS d  ON wr.wr_returned_date_sk = d.d_date_sk
  WHERE d.d_date IN (
    SELECT d2.d_date
    FROM date_dim AS d2
    WHERE d2.d_week_seq IN (
      SELECT d3.d_week_seq
      FROM date_dim AS d3
      WHERE d3.d_date IN (
        DATE '1999-06-30',
        DATE '1999-08-28',
        DATE '1999-11-18'
      )
    )
  )
  GROUP BY i.i_item_id
)
SELECT
  s.item_id,
  s.sr_item_qty,
  s.sr_item_qty  / ((s.sr_item_qty + c.cr_item_qty + w.wr_item_qty)/3.0) * 100 AS sr_dev,
  c.cr_item_qty,
  c.cr_item_qty  / ((s.sr_item_qty + c.cr_item_qty + w.wr_item_qty)/3.0) * 100 AS cr_dev,
  w.wr_item_qty,
  w.wr_item_qty  / ((s.sr_item_qty + c.cr_item_qty + w.wr_item_qty)/3.0) * 100 AS wr_dev,
  (s.sr_item_qty + c.cr_item_qty + w.wr_item_qty)/3.0               AS average
FROM sr_items AS s
JOIN cr_items AS c ON s.item_id = c.item_id
JOIN wr_items AS w ON s.item_id = w.item_id
ORDER BY s.item_id, s.sr_item_qty
LIMIT 100;
