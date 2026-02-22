"""
repeat rate: Birden fazla alışveriş yapan müşteri sayısı / toplam müşteri sayısı
churn rate = 1 - repeat rate
profit margin = total price * 0.10
purchase frequency = total transaction / total num of customers
average order value = total price / total transaction
customer value = average order value * purchase frequency
CLTV = (customer value / churn rate) * profit margin
"""

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.4f" %x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()

df = df[~df["Invoice"].str.contains("C", na=False)]


df.isnull().sum()
df.dropna(inplace=True)

df.describe().T

df = df[(df["Quantity"] > 0)]

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_c = pd.DataFrame()

cltv_c["total_transaction"] = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique()})
cltv_c.head()

cltv_c["total_unit"] = df.groupby("Customer ID").agg({"Quantity": "sum"})

#cltv_c["total_q"] = df.groupby("Customer ID").agg({"Quantity": lambda x: x.sum()})
#cltv_c.drop("total_q", axis=1, inplace=True)

cltv_c["total_price"] = df.groupby("Customer ID").agg({"TotalPrice": "sum"})

#["total_p"] = df.groupby("Customer ID").agg({"TotalPrice": lambda x:x.sum()})
#cltv_c.drop("total_p", axis=1, inplace=True)

cltv_c["avg_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

cltv_c["customer_value"] = cltv_c["avg_order_value"] * cltv_c["purchase_frequency"]

cltv_c["CLTV"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.head()

cltv_c.sort_values(by="CLTV", ascending=False)

cltv_c["segment"] = pd.qcut(cltv_c["CLTV"], 4, labels=["D", "C", "B", "A"])

cltv_c.groupby("segment").agg(["count", "sum", "mean"])

#cltv_c.to_csv("cltvs.csv")
