# Recommendation Systems

This project demonstrates the implementation of different **recommendation system techniques** used in modern platforms such as Netflix, Amazon, and Spotify.

Recommendation systems analyze user behavior and item characteristics to suggest relevant items to users. These systems are widely used in e-commerce, streaming platforms, and social media to improve user experience and engagement.

---

# Project Structure
src/
├── 7-arl_recommender.py
├── 8-hybrid_recommender.py
└── 9-arl_recommender_system.py

---

# 1. Association Rule Learning Recommender

This project builds a recommendation system using **Association Rule Learning (ARL)**.

Association rules discover relationships between items frequently purchased together. These rules are commonly used in **market basket analysis**.

Example:
If a customer buys → Bread
They are also likely to buy → Butter

### Techniques Used

- Apriori Algorithm
- Association Rules
- Support
- Confidence
- Lift

### Goal

Identify **frequently purchased item combinations** and generate product recommendations.

Example Output:
Customers who bought item A are likely to buy item B

---

# 2. Hybrid Recommendation System

Hybrid recommender systems combine multiple recommendation techniques to produce better results.

Most modern recommendation engines combine:

- Collaborative Filtering
- Content-Based Filtering

This approach improves recommendation quality by leveraging both **user behavior patterns and item characteristics**.

### Techniques Used

- Similarity calculations
- User behavior analysis
- Hybrid recommendation logic

### Goal

Generate more accurate recommendations by combining multiple recommendation strategies.

---

# 3. Advanced ARL Recommendation System

This project extends association rule learning to build a **full recommendation pipeline**.

Steps:

1. Data preprocessing
2. Market basket analysis
3. Association rule generation
4. Recommendation function creation

### Output

A recommendation engine that suggests products based on customer purchase history.

Example:
recommend_product("Laptop") → Mouse, Laptop Bag

---

# Technologies Used

- Python
- Pandas
- mlxtend
- Association Rule Learning
- Data Analysis

---

# Key Takeaways

- Recommendation systems improve user experience by suggesting relevant items.
- Association rule learning is widely used in **retail recommendation systems**.
- Hybrid systems combine multiple approaches to increase recommendation accuracy.

---

# Author

Dilek Mirac Colak  
Data Science Portfolio