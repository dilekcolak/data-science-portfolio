import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.4f" %x)

############################## GÖREV 1 ######################################
#Adım 1: flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv("./datasets/flo_data_20k.csv")
df = df_.copy()

#Adım 2:  Aykırı değerleri baskılamak için gerekli olan outlier_thresholdsve replace_with_thresholdsfonksiyonlarını tanımlayınız.
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

#Adım 3:  "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.
cols = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
"customer_value_total_ever_online"]

for col in cols:
    replace_with_thresholds(df, col)

df.describe().T

#Adım 4:  Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["total_purch_count"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_val"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

df.head()

#Adım 5: Değişkentiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()

date_cols = [col for col in df.columns if "_date" in col]

df[date_cols] = df[date_cols].apply(pd.to_datetime)
df.info()

############################## GÖREV 2 ######################################
#Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
today_date = dt.datetime(2021, 6, 1, 00, 00,00)

#Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz. Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).dt.days) / 7
cltv_df["frequency"] = df["total_purch_count"]
cltv_df["monetary_cltv_average"] = df["total_customer_val"] / df["total_purch_count"]

cltv_df.head()

############################## GÖREV 3 ######################################
#Adım 1:BG/NBD modelinifit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

#a) 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["T_weekly"],
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"])
#b) 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_for_6_month"] = bgf.predict(24,
                                               cltv_df["T_weekly"],
                                               cltv_df["frequency"],
                                               cltv_df["recency_cltv_weekly"])

#Adım 2:Gamma-Gamma modelinifit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_average"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_average"])

#Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency_cltv_weekly"],
                                              cltv_df["T_weekly"],
                                              cltv_df["monetary_cltv_average"],
                                              time=6,
                                              freq="W",
                                              discount_rate=0.01)

# Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df.sort_values(by="cltv", ascending=False).head(20)

############################## GÖREV 4 ######################################
#Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])


cltv_df.head(10)

#Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.
cltv_df.groupby("segment").agg({"recency_cltv_weekly": "mean",
                                "frequency": "mean",
                                "monetary_cltv_average": "mean",
                                "exp_sales_3_month": "mean",
                                "exp_average_value": "mean",
                                "cltv": "mean"})

