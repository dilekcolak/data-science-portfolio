"""
Missing Values
-Silerek(dropna)
-Değer atayarak(fillna)
-Tahmine dayalı yöntemler ile doldurarak(fillna)
eksik değerlerden kurtulabiliriz.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.4f" %x)
pd.set_option("display.width", 500)

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

#Eksik değerlerin yakalanması
##############################################

#Dataframe'de eksik değer var mı?
df.isnull().values.any()
#Hangi sütunda kaç eksik değer var?
df.isnull().sum()
#Dataset'teki toplam eksik değer sayısı kaçtır?
df.isnull().sum().sum()
#Sütunlardaki tam değer sayısı kaçtır?
df.notnull().sum()
#Dataset'teki toplam tam değer sayısı kaçtır?
df.notnull().sum().sum()
#En az bir tane eksik değere sahip olan gözlem birimleri nelerdir?
df[df.isnull().any(axis=1)]

df.isnull().sum().sort_values(ascending=False)

#Değişkenlerdeki eksik değerlerin oranı nedir?
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

#Eksik değere sahip sütunları nasıl belirleriz?
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

##############################################################
#Eksik değerlerden kurtulmak
##############################################################
df.shape

#ÇÖZÜM 1: Hızlıca silmek
##############################
df.dropna().shape

#ÇÖZÜM 2: Basit atama yöntemleri ile doldurmak
########################################################
df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull()

df["Age"].fillna(0).isnull().sum()

#Kategorik değişkenlerde hata verir
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

#Kategorik değişkenler için en yaygın kullanılan yöntem mod almaktır. Sütunda en çok tekrar eden değer ile eksik değer doldurulur.
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

#String bir değer ile de doldurabiliriz
df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

#ÇÖZÜM 3: Kategorik değişken kırılımında değer atamak
################################################################
df.groupby("Sex")["Age"].mean()
df["Age"].mean()
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = (df["Sex"] == "female").mean()
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()
df = load()

#ÇÖZÜM 4: Tahmine dayalı atama ile doldurma
################################################################
"""
One-Hot-Encoding -> modelin beklediği belirli bir standart vardır. 
Bu standartlara uymak için değişkenleri düzenlememiz gerekiyor.

Standartlaştırma -> KNN uzaklık temelli bir algoitmadır.
Değişkenlerin etkisinin eşit olması için standartlaştırma yapmamız gerekir.
"""

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and
                   dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and
                col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {cat_cols}")
    print(f"num_cols: {num_cols}")
    print(f"num_but_cat: {num_but_cat}")
    print(f"cat_but_car: {cat_but_car}")
    return cat_cols, num_cols, cat_but_car

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

#One-hot-encoder uygulayalım
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

#drop_first argümanı ile alfabetik olarak ilk gelen sınıfı atar.

dff.head()
df.head()
df["Embarked"].value_counts()

#Değişkenleri standartlaştıralım
#Değişkenleri 0 ile 1 arasına sıkıştırır
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

#fit_transform metodu numpy array döndürür. Bu yüzden dataframe dönüşümüne sokuyoruz.

#KNN yöntemini uygulayalım
#En yakın 5 komşusunun yaş ortalamasına göre değer ataması yapar.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

#Karşılaştırma yapabilmek için standartlaştırma işlemini geri almamız gerekiyor.
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

#NASIL TAKİP EDECEĞİZ?
df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]

#GELİŞMİŞ ANALİZLER YAPALIM
"""
Eksik değerlerde bağımlılık var mı?

iki tip bağımlılık vardır:
    1.Kredi kartı kullanmayan birinin kredi_kartı_var_mı bilgisi no olduğu için harcama bilgisinin eksik olması.
    2.Birden fazla değişkenin birlikte eksik veya tam olması durumu.
"""

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

#Eksik değerlerin bağımlı değişken ile ilişkisini inceleyelim.
def missing_table(dataframe, na_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (100 * dataframe[na_cols].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_cols

missing_table(df)
na_cols = missing_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_cols:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Survived", na_cols)


