-- Basic summary statistics
SELECT
    COUNT(*) AS total_records,
    AVG(salary) AS avg_salary,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary
FROM employees;
