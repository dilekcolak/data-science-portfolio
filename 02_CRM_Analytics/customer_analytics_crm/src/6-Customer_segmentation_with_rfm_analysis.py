import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.4f" %x)

############################## GÖREV 1 ######################################
#Adım 1: Online Retail II excelindeki 2010-2011 verisini okuyunuz. Oluşturduğunuz dataframe’in kopyasını oluşturunuz.
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

#Adım 2: Veri setinin betimsel istatistiklerini inceleyiniz.
df.head()
df.info()
df.describe().T

#Adım 3: Veri setinde eksik gözlem var mı? Varsa hangi değişkende kaç tane eksik gözlem vardır?
df.isnull().sum()
#Description ve Customer ID değişkenlerinde eksik değerler mevcuttur.

#Adım 4: Eksik gözlemleri veri setinden çıkartınız. Çıkarma işleminde ‘inplace=True’ parametresini kullanınız.
df.dropna(inplace=True)

#Adım 5: Eşsiz ürün sayısı kaçtır?
df["StockCode"].nunique()

#Adım 6: Hangi üründen kaçar tane vardır?
df["StockCode"].value_counts()

#Adım 7: En çok sipariş edilen 5 ürünü çoktan aza doğru sıralayınız
df.groupby("Description").agg({"Quantity": "sum"}).sort_values(by="Quantity", ascending=False).head()

#Adım 8: Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.
df = df[~df["Invoice"].str.contains("C", na=False)]

#Adım 9: Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz
df["TotalPrice"] = df["Quantity"] * df["Price"]


############################## GÖREV 2 ######################################
#Adım 1: Recency, Frequency ve Monetarytanımlarını yapınız. Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
rfm = pd.DataFrame()
today_date = dt.datetime(2011, 12, 11)

df.head()

rfm["recency"] = df.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days})

rfm["frequency"] = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique()})

rfm["monetary"] = df.groupby("Customer ID").agg({"TotalPrice": "sum"})

rfm = rfm[rfm["monetary"] > 0]

rfm.head()

############################## GÖREV 3 ######################################
#Adım 1-2: Recency, Frequency ve Monetarymetriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz. Bu skorları recency_score, frequency_scoreve monetary_scoreolarak kaydediniz.
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

#Adım 3: recency_scoreve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
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

rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean"}).head()

loyal_customers = pd.DataFrame()

loyal_customers["CustomerID"] = rfm[rfm["segment"] == "loyal_customers"].index.astype(int)

#Adım 3: "LoyalCustomers" sınıfına ait customer ID'leri seçerek excel çıktısını alınız.
loyal_customers.to_csv("datasets/loyal_customers.csv")
