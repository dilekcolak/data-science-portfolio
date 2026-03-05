"""
FEATURE EXTRACTION:
    Ham veriden değişken türetiriz.
        1- Yapısal verilerden değişkenler türetmek
        2- Yapısal olmayan (görüntü, ses, yazı) verilerden değişkenler türetmek

    tarih ve saatten oluşan bir değişken sütunundan yıl, ay, gün, saat, haftanın günü gibi
    değişkenler türetilebilir.

    Binary Features: Flag, bool, True-False
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option("display.float_format", lambda x: "%.2f" %x)
pd.set_option("display.width", 500)

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()

#BINARY FEATURES

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int") #NaN ise 0, dolu ise 1

df.groupby("NEW_CABIN_BOOL")["Survived"].mean()

df.head()

#bu oran farkını anlamlı mı yoksa rastgele mi oluşmuş? Bunu anlamak için oran testi yapacağız.
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print("Test Stat = %.4f, p-value= %.4f" % (test_stat, pvalue))

#H0 hipotezi neydi oranlar arasında fark yoktur der. Yani p1 = p2
#pvalue değeri 0.05'ten küçük olduğu için bu hipotez reddedilir.

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE")["Survived"].mean()

#TEXT'LER ÜZERİNDEN ÖZELLİK TÜRETMEK

#1-letter count
df["NEW_NAME_COUNT"] = df["Name"].str.len()
df.head()

#2-word count
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

#3-Özel yapıları yakalamak
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR")["Survived"].mean()

df["NEW_TITLE"] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df.head()

df[["NEW_TITLE", "Survived", "Age"]].groupby("NEW_TITLE").agg({"Survived": "mean", "Age": ["count", "mean"]})

#DATE DEĞİŞKENLERİ ÜRETMEK
dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"])

dff["Year"] = dff["Timestamp"].dt.year

dff["Month"] = dff["Timestamp"].dt.month

dff["year_diff"] = date.today().year - dff["Year"]

dff["month_diff"] = (date.today().year - dff["Year"]) * 12 + date.today().month - dff["Month"]

dff["day_name"] = dff["Timestamp"].dt.day_name()

#Özellik etkileşimleri
df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

