# Measurement Problems

This project focuses on solving common **measurement problems in data science**, including experiment evaluation and product ranking strategies.

The repository contains two main analyses:

- A/B Testing
- Product Rating & Sorting (Amazon Reviews)

These techniques are widely used in **data-driven decision making**, **product optimization**, and **recommendation systems**.

---

# Project Structure
src/

├── 7-AB_Testi.py

└── 8-rating_sorting_amazon.py

---

# 1. A/B Testing

A/B testing is an experimental method used to compare two versions of a product, feature, or interface to determine which performs better based on statistical evidence. :contentReference[oaicite:0]{index=0}

In this project:

- Control and test groups are compared
- Statistical hypothesis testing is performed
- Metrics such as conversion rate or engagement are evaluated
- Significance of differences is analyzed

### Key Concepts

- Hypothesis testing
- Control vs Test group
- Statistical significance
- Confidence level
- p-value interpretation

### Goal

Determine whether a **new product version or feature creates a statistically significant improvement**.

---

# 2. Product Rating & Sorting (Amazon Case)

Online platforms such as Amazon need effective ways to rank products based on user feedback.

A simple average rating is often misleading because:

- Products with very few reviews may appear highly rated
- Older reviews may not reflect current product quality

Amazon therefore uses **weighted and machine-learning based rating systems**, considering factors like review recency, verified purchases, and helpful votes. :contentReference[oaicite:1]{index=1}

### Methods Implemented

This project explores different ranking techniques such as:

- Average Rating
- Weighted Rating
- Time-Based Weighted Average
- Up/Down Vote Scoring
- Wilson Lower Bound Score

### Goal

Improve product ranking reliability by incorporating:

- review recency
- review counts
- helpful vote ratios

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scipy
- Statistical Analysis

---

# Key Takeaways

- Data-driven experiments are critical for product optimization.
- A/B testing allows businesses to validate decisions with statistical evidence.
- Proper ranking algorithms improve recommendation quality and user trust.

---

# Author

Dilek Mirac Colak  
Data Science Portfolio