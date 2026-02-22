"""
BG-NBD ve Gamma-Gamma ile CLTV Tahmini

İş Problemi:
İngiltere merkezli perakende şirketi satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir. Şirketin
orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin
tahmin edilmesi gerekmektedir

Görev 1:  BG-NBD ve Gamma-Gamma Modellerini Kurarak 6 Aylık CLTV Tahmini Yapılması
    Adım 1: 2010-2011 yıllarındaki veriyi kullanarak İngiltere’deki müşteriler için 6 aylık CLTV tahmini yapınız.
    Adım 2: Elde ettiğiniz sonuçları yorumlayıp, değerlendiriniz.
Görev 2:  Farklı Zaman Periyotlarından Oluşan CLTV Analizi
    Adım 1: 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
    Adım 2: 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
    Adım 3: Fark var mı? Varsa sizce neden olabilir?
Görev 3:  Segmentasyon ve Aksiyon Önerileri
    Adım 1: 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
    Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.
"""

import datetime as dt
import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" %x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

def outlier_thresholds(dataframe, value, q1=0.01, q3=0.99):
    quartile1 = dataframe[value].quantile(q1)
    quartile3 = dataframe[value].quantile(q3)
    iqr = q3 - q1
    min_value = quartile1 - 1.5 * iqr
    max_value = quartile3 + 1.5 * iqr

    min_value = int(np.floor(min_value))
    max_value = int(np.ceil(max_value))

    return min_value, max_value

def replace_with_thresholds(dataframe, value, q1=0.01, q3=0.99):
    low, up = outlier_thresholds(dataframe, value, q1, q3)
    dataframe.loc[dataframe[value] > up, value] = up
    dataframe.loc[dataframe[value] < low, value] = low

df.head()
df["Invoice"].value_counts()

df = df[~df["Invoice"].str.contains("C", na=False)]

df.describe().T

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Price"] * df["Quantity"]

df["InvoiceDate"].max()
today = dt.datetime(2011, 12, 11)

cltv_df = pd.DataFrame()

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today - date.min()).days],
                                         "Invoice": lambda invoice: invoice.nunique(),
                                         "TotalPrice": lambda price: price.sum()})
cltv_df.columns = ["recency", "tenor", "frequency", "total_price"]

cltv_df["monetary"] = cltv_df["total_price"] / cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["tenor"] = cltv_df["tenor"] / 7

cltv_df.shape
cltv_df.head()

betaGeoFitter = BetaGeoFitter(penalizer_coef=0.001)
betaGeoFitter.fit(cltv_df["frequency"],
                  cltv_df["recency"],
                  cltv_df["tenor"])

betaGeoFitter.conditional_expected_number_of_purchases_up_to_time(6*4,
                                                                  cltv_df["frequency"],
                                                                  cltv_df["recency"],
                                                                  cltv_df["tenor"]).sort_values(ascending=False)

gammaGammaFitter = GammaGammaFitter(penalizer_coef=0.001)
gammaGammaFitter.fit(cltv_df["frequency"],
                     cltv_df["monetary"])

gammaGammaFitter.conditional_expected_average_profit(cltv_df["frequency"],
                                                     cltv_df["monetary"])

cltv_df["cltv"] = gammaGammaFitter.customer_lifetime_value(betaGeoFitter,
                                         cltv_df["frequency"],
                                         cltv_df["recency"],
                                         cltv_df["tenor"],
                                         cltv_df["monetary"],
                                         time=6,
                                         freq="W",
                                         discount_rate=0.01)

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.reset_index(inplace=True)
cltv_df.sort_values(by="cltv", ascending=False)
cltv_df.groupby("segment").agg(["min", "mean", "max"])["cltv"]

######################################GÖREV 2####################################
uk_df = cltv_df.merge(df[["Customer ID", "Country"]], how="left", on="Customer ID")
uk_df.head()
uk_df.shape

uk_df["1_month_cltv"] = gammaGammaFitter.customer_lifetime_value(betaGeoFitter,
                                         uk_df["frequency"],
                                         uk_df["recency"],
                                         cltv_df["tenor"],
                                         cltv_df["monetary"],
                                         time=1,
                                         discount_rate=0.01)


uk_df["12_month_cltv"] = gammaGammaFitter.customer_lifetime_value(betaGeoFitter,
                                         uk_df["frequency"],
                                         uk_df["recency"],
                                         cltv_df["tenor"],
                                         cltv_df["monetary"],
                                         time=12,
                                         discount_rate=0.01)

best_of_1_month_cltv = uk_df["1_month_cltv"].sort_values(ascending=False).head(10).index
best_of_12_month_cltv = uk_df["12_month_cltv"].sort_values(ascending=False).head(10).index