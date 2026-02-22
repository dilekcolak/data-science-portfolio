import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.4f" %x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")

df = df_.copy()

df.head()
df.shape

df.isnull().sum()
df.dropna(inplace=True)

df["Description"].nunique()

df.groupby(["Customer ID", "Description"])["Quantity"].sum()
df.groupby(["Customer ID", "Description"])["Quantity"].sum().sort_values(ascending=False).head()

df["TotalPrice"] = df["Quantity"] * df["Price"]

df.groupby("Invoice").agg({"TotalPrice": "sum"})
df.groupby("Customer ID").agg({"TotalPrice": "sum"})

df.describe().T

df["Invoice"].nunique()

df = df[~df["Invoice"].str.contains("C", na=False)]

df.info()

df["InvoiceDate"].max()

today = dt.datetime(2010, 12, 11)
type(today)

rfm = pd.DataFrame()
rfm["recency"] = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today - date.max()).days})

rfm.head()

#Aşağıdaki yöntem ilk akla gelir ama bu eşsiz değerleri ayrıştırmaz. O yüzden bir sonraki adıma bakınız.
#rfm["fr"] = df.groupby("Customer ID").agg({"Invoice": "count"})
#rfm.drop("fr", axis=1, inplace=True)

rfm["frequency"] = df.groupby("Customer ID").agg({"Invoice": lambda num: num.nunique()})

rfm["monetary"] = df.groupby("Customer ID").agg({"TotalPrice": "sum"})

rfm = rfm[rfm["monetary"] > 0]

rfm.describe().T

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

rfm["rf_score"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

seg_map = {
        r'[1-2][1-2]': "hibernating",
        r'[1-2][3-4]': "at_Risk",
        r'[1-2][5]': "cant_loose",
        r'[3][1-2]': "about_to_sleep",
        r'[3][3]': "need_attention",
        r'[3-4][4-5]': "loyal_customers",
        r'[4][1]': "promising",
        r'[5][1]': "new_customers",
        r'[4-5][2-3]': "potential_loyalist",
        r'[5][4-5]': "champions",
    }

rfm["segment"] = rfm["rf_score"].replace(seg_map, regex=True)

rfm.head()

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

rfm.groupby("segment").agg({"recency": ["mean", "count"],
                            "frequency": ["mean", "count"],
                            "monetary": ["mean", "count"]})

rfm[rfm["segment"] == "need_attention"].head()

rfm[rfm["segment"] == "need_attention"].index

new_df = pd.DataFrame()

new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index.astype(int)

new_df.head()

new_df.to_csv("new_customers.csv")

rfm.to_csv("rfm.csv")

