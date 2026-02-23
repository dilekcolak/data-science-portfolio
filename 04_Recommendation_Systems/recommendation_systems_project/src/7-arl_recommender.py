import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import datetime as dt

#Adım 1: armut_data.csv dosyasını okutunuz
df = pd.read_csv("datasets/armut_data.csv")

#Veriyi inceleyelim
df.head()
df.shape
df.describe().T
df.info()
df.isnull().sum()

############################################################################
#GÖREV 1: VERİYİ HAZIRLAMA
############################################################################

#Adım 2: ServiceID ve CategoryID’yi "_"  ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.
df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

#Adım 3: Sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])

df["NewDate"] = df["CreateDate"].dt.to_period("M")

df["SepetID"] = df["UserId"].astype(str) + "_" + df["NewDate"].astype(str)

###############################################################################
#GÖREV 2: BİRLİKTELİK KURALLARI ÜRETME VE ÖNERİ YAPMA
###############################################################################

#Adım 1: Sepet ve Hizmet için pivot table oluşturun
basket_df = df.pivot_table(index="SepetID", columns="Hizmet", values="UserId", aggfunc="count")

basket_df.fillna(0, inplace=True)

basket_df = basket_df.applymap(lambda x: 0 if x == 0 else 1)

#Başka bir yol
basket_df = df.groupby(["SepetID", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)


#Adım 2: Birliktelik kurallarını oluşturunuz.

def create_rules(dataframe, metric="support", min_support=0.01, min_th=0.01):
    frequent_itemsets = apriori(dataframe, min_support=min_support, use_colnames=True)
    arl_rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_th)
    return arl_rules

rules = create_rules(basket_df)

rules.info()

#Adım 3:arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
def arl_recommender(rules, target, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, antecedents in enumerate(sorted_rules["antecedents"]):
        if target in antecedents:
            for con in sorted_rules.iloc[i]["consequents"]:
                if con not in recommendation_list:
                    recommendation_list.append(con)
    return recommendation_list[:rec_count]

rec_list = arl_recommender(rules, "2_0", rec_count=5)