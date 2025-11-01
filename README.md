# INIT_TASK

## Project Summary

Hi everyone, my name is Aryan Patil and I have created this project which performs end-to-end exploratory data analysis (EDA) and customer behavior analytics on a retail dataset consisting of:

1. Customers

2. Products

3. Transactions

## Key Steps Performed

Loaded and inspected datasets (shape, schema, null values, duplicates)

Cleaned missing values and handled data types (date parsing, numeric fixes)

Performed feature engineering on transactions:

  a. Day, month, hour from timestamps

  b. Total spend per order

  c. Derived metrics (purchase frequency, revenue, etc.)

Merged datasets into a unified view

Generated customer segmentation metrics (RFM analysis)

Calculated product-level KPIs (revenue, quantity sold)

Identified suspicious/irregular transactions

Created customer lifetime value (CLV) calculation

Measured performance for large-data filtering operations

The analysis gives a full picture of customer behavior, revenue patterns, product performance, and potential fraud detection.

## Dependencies / Libraries Used

| Library                   | Purpose                                |
| ------------------------- | -------------------------------------- |
| `numpy`                   | Numerical computations                 |
| `pandas`                  | Data manipulation, cleaning, EDA       |
| `time`                    | Execution-time measurement             |
| `datetime` *(via pandas)* | Date-time parsing & feature extraction |
