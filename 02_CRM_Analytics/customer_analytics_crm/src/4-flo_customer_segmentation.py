import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.4f" %x)

############################## GÖREV 1 ######################################
#Adım 1:  flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()

#Adım 2: Veri setinde ilk 10 gözlem, değişken isimleri, betimsel istatistik, boş değer, değişken tipleri incelemesi yapınız.
df.head(10)

df["order_channel"].value_counts()
df["last_order_channel"].value_counts()

df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()

#Adım 3: Omnichannel müşterilerin hem online'dan hemdeoffline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

#Adım 4: Değişkentiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_cols = [col for col in df.columns if "_date" in col]

for col in date_cols:
    df[col] = pd.to_datetime(df[col])

df.info()

#Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({"master_id": "count",
                                 "total_order_num": "sum",
                                 "total_customer_value": "sum"})

#Adım 6:  En fazla kazancı getiren ilk 10 müşteriyi sıralayınız
df.sort_values(by="total_customer_value", ascending=False).head(10)

#Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız
df.sort_values(by="total_order_num", ascending=False).head(10)

#Adım 8: Veri önhazırlık sürecini fonksiyonlaştırınız
def eda(df):
    df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

    df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    # Adım 4
    date_cols = [col for col in df.columns if "_date" in col]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    df.groupby("order_channel").agg({"master_id": "count",
                                     "total_order_num": "sum",
                                     "total_customer_value": "sum"})

    print("Parasal getirisi en yüksek olan 10 müşteri:", df.sort_values(by="total_customer_value", ascending=False).head(10).index)

    print("En fazla alışveriş yapan 10 müşteri: ", df.sort_values(by="total_order_num", ascending=False).head(10).index)

############################## GÖREV 2 ######################################
#Adım 1
#Recency: Müşterinin en son alışveriş yaptığı tarihten bu güne kadar geçen süre
#Frequency: Müşterinin yaptığı alışveriş sayısı
#Monetary: Müşterinin yaptığı alışverişler sonucu yaptığı toplam ödeme

#Adım 2: Recency, Frequency ve Monetary tanımlarını yapınız.
rfm = pd.DataFrame()
today = dt.datetime.today()
rfm["Recency"] = (today - df["last_order_date"]).dt.days
rfm["Frequency"] = df["total_order_num"].astype(int)
rfm["Monetary"] = df["total_customer_value"].astype(int)

rfm.index = df.master_id
rfm.head()

############################## GÖREV 3 ######################################
#Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

#Adım 2: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

############################## GÖREV 4 ######################################
#Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
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

#Adım 2: seg_map yardımı ile skorları segmentlere çeviriniz.
rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

############################## GÖREV 5 ######################################
#Adım 1:  Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").mean()

#Adım 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.
rfm.reset_index(inplace=True)
df_final = df.merge(rfm, on="master_id", how="left")

#a)FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
#tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
#iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
#yapankişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz
loyal_woman_customers_df = df_final[(df_final["segment"].isin(["champions", "loyal_customers"])) &
                               (df_final["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]

loyal_woman_customers_df.to_csv("flo_kadin_markası_hedef_müsteriler.csv", index=False)

#b)Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
#iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
#gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.
discount_customers = df_final[(df_final["segment"].isin(["cant_loose", "atrisk", "hibernating", "new_customers"])) &
                              (df_final["interested_in_categories_12"].str.contains("ERKEK|ÇOCUK", na=False))]["master_id"]

discount_customers.to_csv("hedef_indirim_müsterileri.csv", index=False)

df.head()
df.columns
df_final.head()
rfm.head()
rfm.columns
