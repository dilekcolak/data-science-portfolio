import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" %x)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)

    iqr = quartile3 - quartile1

    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr

    low_limit = int(np.floor(low_limit))
    up_limit = int(np.ceil(up_limit))

    return up_limit, low_limit

def replace_with_thresholds(dataframe, variable):
    up, low = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > up, variable] = up
    dataframe.loc[dataframe[variable] < low, variable] = low

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()

df.describe().T

df.isnull().sum()
df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")

replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

df["InvoiceDate"].max()

today = dt.datetime(2011, 12, 11)

cltv_df = pd.DataFrame()

cltv_df["recency"] = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (date.max() - date.min()).days})
cltv_df["tenor"] = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today - date.min()).days})
cltv_df["frequency"] = df.groupby("Customer ID").agg({"Invoice": lambda invoice: invoice.nunique()})
cltv_df["total_price"] = df.groupby("Customer ID").agg({"TotalPrice": "sum"})
cltv_df["monetary"] = cltv_df["total_price"] / cltv_df["frequency"]

cltv_df = cltv_df[cltv_df["frequency"] > 1]

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["tenor"] = cltv_df["tenor"] / 7

cltv_df.head()

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["tenor"])

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["tenor"]).sort_values(ascending=False)

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["tenor"])

cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                       cltv_df["frequency"],
                                       cltv_df["recency"],
                                       cltv_df["tenor"])
expected_total_transaction_for_1_month = bgf.predict(4, cltv_df["frequency"], cltv_df["recency"], cltv_df["tenor"]).sum()

cltv_df.head()

plot_period_transactions(bgf)
plt.show()

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"]).sort_values(ascending=False).head()

cltv_df["expected_avg_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df["expected_avg_profit"].sort_values(ascending=False).head()

cltv_df.sort_values(by="expected_avg_profit", ascending=False).head(20)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["tenor"],
                                   cltv_df["monetary"],
                                   time=3,
                                   freq="W",
                                   discount_rate=0.01)

cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head(20)

cltv_final.groupby("segment").agg(["sum", "mean", "count"])

