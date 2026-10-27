-- QUERY Q1 WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS qualifying_spend
FROM orders
WHERE status='completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- QUERY Q2 city and category revenue
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue_inr
FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id)
WHERE o.status='completed'
GROUP BY c.city,p.category
ORDER BY c.city,p.category;

-- QUERY Q3 products above average catalogue price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- QUERY Q4 customers with no completed order
SELECT c.customer_id, c.city
FROM customers c LEFT JOIN orders o
ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- QUERY Q5 customers above average completed spend
WITH spend AS (
  SELECT o.customer_id, SUM(o.order_amount) AS total_spend
  FROM orders o
  WHERE o.status='completed'
  GROUP BY o.customer_id
), avg_spend AS (
  SELECT AVG(total_spend) AS avg_total FROM spend
)
SELECT spend.customer_id, spend.total_spend
FROM spend CROSS JOIN avg_spend
WHERE spend.total_spend > avg_spend.avg_total
ORDER BY spend.customer_id;

-- QUERY Q6 ranking window functions by city
WITH spend AS (
  SELECT c.customer_id,c.city,SUM(o.order_amount) AS total_spend
  FROM customers c JOIN orders o USING(customer_id)
  WHERE o.status='completed'
  GROUP BY c.customer_id,c.city
)
SELECT customer_id,city,ROUND(total_spend,2) AS total_spend,
 RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS rank_value,
 DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_value,
 ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC,customer_id) AS row_number_value
FROM spend
ORDER BY city, row_number_value;

-- QUERY Q7 daily running total
WITH daily AS (
 SELECT substr(order_ts,1,10) AS day, SUM(order_amount) AS revenue
 FROM orders WHERE status='completed'
 GROUP BY substr(order_ts,1,10)
)
SELECT day, ROUND(revenue,2) AS revenue,
 ROUND(SUM(revenue) OVER(ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),2) AS cumulative_revenue
FROM daily ORDER BY day;
