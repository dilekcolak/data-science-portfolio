####################################################################
#ASSOCIATION RULE BASED RECOMMENDER SYSTEM
####################################################################
"""
İŞ PROBLEMİ:
Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir. Bu sepet bilgilerine en
uygun ürün önerisini birliktelik kuralı kullanarak yapınız. Ürün önerileri 1 tane
ya da 1'den fazla olabilir. Karar kurallarını 2010-2011 Germany müşterileri
üzerinden türetiniz.

Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
Kullanıcı 2’nin sepetinde bulunan ürünün id'si: 23235
Kullanıcı 3’ün sepetinde bulunan ürünün id'si: 22747

VERİ SETİ HİKAYESİ:
Online retail II isimli veri seti İngiltere merkezli bir perakende şirketinin
01/12/2009 - 09/12/2011 tarihleri arasındaki online satış işlemlerini içeriyor.
Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterinin
toptancı olduğu bilgisi mevcuttur.

InvoiceNo: Fatura numarası (C ile başlayanlar iade işlemleri ifade eder)
StockCode: Ürün kodu
Description: Ürün adı
Quantity: Ürün adedi (Faturadaki ürünlerden kaçar tane satıldığı)
InvoiceDate: Fatura tarihi
UnitPrice: Fatura fiyatı(Sterlin)
CustomerID: Eşsiz müşteri numarası
Country: Ülke adı
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

#GÖREV 1: Veriyi Hazırlama
#######################################################################################################

#Adım 1: Online Retail II veri setinden 2010-2011 sheet'ini okutunuz.
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.shape
df.info()
df.describe().T

#Adım 2: StockCode'u POST olan gözlem birimlerini drop ediniz. (POST: her faturaya eklenen bedeldir, ürünü ifade etmez)
posts = df[df["StockCode"] == "POST"]
df = df.drop(posts.index)

#Başka yol
df = df[df["StockCode"] != "POST"]

#Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
df.isnull().sum()
df = df.dropna()

#Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız.
df = df[~df["Invoice"].str.contains("C", na=False)]

#Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
df = df[df["Price"] >= 0]

#Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.
def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    iqr = quartile3 - quartile1

    low = quartile1 - 1.5 * iqr
    up = quartile3 + 1.5 * iqr

    return low, up

def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low, up = outlier_threshold(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].empty:
        print(col_name, " has no any outliers.")
    else:
        print(col_name, "has outliers.")

def replace_outliers_with_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    low, up = outlier_threshold(dataframe, col_name, q1, q3)

    dataframe.loc[dataframe[col_name] < low, col_name] = low
    dataframe.loc[dataframe[col_name] > up, col_name]= up

check_outlier(df, "Price")
check_outlier(df, "Quantity")

replace_outliers_with_thresholds(df, "Price")
replace_outliers_with_thresholds(df, "Quantity")

#GÖREV 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
#######################################################################################################
germany_df = df[df["Country"] == "Germany"]

germany_df.head()
germany_df.shape

#Adım 1: Fatura ürün pivot table'i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.
def create_invoice_product_df(dataframe):
    return dataframe.pivot_table(index=["Invoice"], columns=["Description"], values="Quantity").fillna(0).applymap(lambda x: 1 if x > 0 else 0)

invoice_product_df = create_invoice_product_df(germany_df)

#Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kuralları bulunuz.

def create_rules(dataframe, metric="support", min_support=0.01 ,min_th=0.01):
    frequent_itemsets = apriori(dataframe, min_support=min_support, use_colnames=True)
    arl_rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_th)
    return arl_rules

arl_rules = create_rules(invoice_product_df)

#GÖREV 3: Sepet İçerisindeki Ürün ID'leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
#######################################################################################################

productID1 = 21987
productID2 = 23235
productID3 = 22747

#Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.
def check_id(dataframe, productID):
    productId = str(productID)
    match = dataframe[dataframe["StockCode"].astype(str) == productId]["Description"]
    if match.empty:
        print(productId, " ID'li ürün bulunamadı.")
        return None
    product_name = match.iloc[0]
    print(productId, ":", product_name)
    return product_name

product_name1 = check_id(germany_df, productID1)
product_name2 = check_id(germany_df, productID2)
product_name3 = check_id(germany_df, productID3)

#Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.
def arl_recommender(rules, product_name, rec_count=1):
    sorted_rules = rules.sort_values(by="lift", ascending=False)
    recommended_products = []

    for i, product in enumerate(sorted_rules["antecedents"]):
        if product_name in product:
            for rec_product in list(sorted_rules["consequents"].iloc[i]):
                if rec_product not in recommended_products:
                    recommended_products.append(rec_product)
    return recommended_products[: rec_count]


#Adım 3: Önerilecek ürünlerinn isimlerine bakınız.
recommended_products1 = arl_recommender(arl_rules, product_name1)
recommended_products2 = arl_recommender(arl_rules, product_name2, 3)
recommended_products3 = arl_recommender(arl_rules, product_name3, 3)