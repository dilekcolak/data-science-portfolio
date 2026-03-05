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
df.head()

#Cabin değişkenlerinde NaN değerlere 0 diğerlerine 1 yazalım
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                               df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                      nobs= [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])
print("pvalue:", pvalue)

#Kişinin yalnız olma veya ailesiyle olma durumuna uygunn bir değişken türetelim
df.loc[df["SibSp"] + df["Parch"] > 0, "NEW_IS_ALONE"] = "NO"
df.loc[df["SibSp"] + df["Parch"] == 0, "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                               df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],
                                      nobs= [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])
print("pvalue:", pvalue)


##############################################################
#Textler üzerinden özellik türetmek
##############################################################
df.head()

#Letter Count
#############################
df["NEW_NAME_COUNT"] = df["Name"].str.len()

#Word Count
#############################
df["NEW_WORD_COUNT"] = df["Name"].apply(lambda x: len(x.split(" ")))

#Özel Yapıları Ayırmak
###############################
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df["NEW_NAME_DR"].value_counts()

###########################################################
#Regex ile değişken türetmek
###########################################################
df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})
df.head()

###################################################
#Date değişkenleri türetmek
###################################################
dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"])

dff["year"] = dff["Timestamp"].dt.year

dff["month"] = dff["Timestamp"].dt.month

dff["year_diff"] = date.today().year - dff["year"]

dff["month_diff"] = (date.today().year - dff["Timestamp"].dt.year) * 12 + (date.today().month - dff["Timestamp"].dt.month)

dff["day_name"] = dff["Timestamp"].dt.day_name()

###############################################################
#Feature Interactions (Özellik etkileşimleri)
###############################################################
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "male") & ((df["Age"] > 21) & (df["Age"] <= 50)), "NEW_SEX_CAT"] = "maturemale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniormale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"

df.loc[(df["Sex"] == "female") & ((df["Age"] > 21) & (df["Age"] <= 50)), "NEW_SEX_CAT"] = "maturefemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.head()
df["NEW_SEX_CAT"].value_counts()

df.groupby("NEW_SEX_CAT")["Survived"].mean()