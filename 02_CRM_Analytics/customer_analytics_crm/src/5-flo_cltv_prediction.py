import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.4f" %x)

############################## GÖREV 1 ######################################
#Adım 1
df_ = pd.read_csv("./datasets/flo_data_20k.csv")
df = df_.copy()

#Adım 2
def outlier_threshold(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    iqr = q3 - q1
    low_limit = round(q1 - 1.5 * iqr)
    upper_limit = round(q3 + 1.5 * iqr)

    return low_limit, upper_limit

def replace_with_thresholds(dataframe, variable):
    low, up = outlier_threshold(dataframe, variable)
    df.loc[(dataframe[variable] > up), variable] = up
    df.loc[(dataframe[variable] < low), variable] = low

df.head()
df.describe().T

#Adım 3
cols = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
"customer_value_total_ever_online"]

for col in cols:
    replace_with_thresholds(df, col)

df.describe().T

#Adım 4
df["total_purch_count"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_val"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

df.head()

#Adım 5
df.info()

date_cols = [col for col in df.columns if "_date" in col]

df[date_cols] = df[date_cols].apply(pd.to_datetime)
df.info()

############################## GÖREV 2 ######################################
#Adım 1
today_date = dt.datetime(2021, 6, 1, 00, 00,00)

#Adım 2
cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).dt.days) / 7
cltv_df["frequency"] = df["total_purch_count"]
cltv_df["monetary_cltv_average"] = df["total_customer_val"] / df["total_purch_count"]

cltv_df.head()

############################## GÖREV 3 ######################################
#Adım 1
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["T_weekly"],
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"])

cltv_df["exp_sales_for_6_month"] = bgf.predict(24,
                                               cltv_df["T_weekly"],
                                               cltv_df["frequency"],
                                               cltv_df["recency_cltv_weekly"])

#Adım 2
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_average"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_average"])

#Adım 3
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency_cltv_weekly"],
                                              cltv_df["T_weekly"],
                                              cltv_df["monetary_cltv_average"],
                                              time=6,
                                              freq="W",
                                              discount_rate=0.01)

cltv_df.sort_values(by="cltv", ascending=False).head(20)

############################## GÖREV 4 ######################################
#Adım 1
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.head(10)

#Adım 2
cltv_df.groupby("segment").agg({"recency_cltv_weekly": "mean",
                                "frequency": "mean",
                                "monetary_cltv_average": "mean",
                                "exp_sales_3_month": "mean",
                                "exp_average_value": "mean",
                                "cltv": "mean"})

