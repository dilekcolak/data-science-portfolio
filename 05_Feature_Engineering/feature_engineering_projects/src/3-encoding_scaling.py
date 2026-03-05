"""
Encoding: Değişkenlerin temsil şekillerinin değiştirilmesidir.

    Label Encoding: Kategorik değişkenin sınıflarının sayısal olarak temsil edilmesidir.
                    Alfabetik olarak sıralama yapar.
                    Eğitim durumu değişkeninde kategoriler arası bir sıralama olduğu için label encoding yapmak mantıklı olur.
                    Futbol takımları değişkeninde kategoriler arasında bir sıralama olmadığı için label encoding yapmak yanıltıcı olur.
                    NEDEN? çünkü model sayısal değerleri büyüklük olarak algılar.
    Binary Encoding: Label encoding ile aynı işi yapar sadece iki sınıflı değişkenler için özelleştirilmiştir.
    One-Hot-Encoding: Label encoding yapamayacağımızı söylediğimiz kategorik değişkenler vardı ya, futbol takımları gibi siniflar arasında herhangi bir sıralama olmamaıs udurmmunda kullanılır.
                    Kategorik değişkenin her bir sınıfını yeni bir değişken olarak atar ve 1-0 olarak kodlar.
                    Dummy değişken tuzağına düşmemek için drop_first=True yapılır.
    Rare Encoding: Çok az sayıda gözlemde görünen sınıfları tek bir sınıf olarak tanımlarız.
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
df.head()
df["Sex"].head()

#Label Encoding
######################################################
le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]

#Labelların orijinal değerlerini görmek için
le.inverse_transform([1, 0])

def label_encoder(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return dataframe

df = label_encoder(df, "Sex")
df.head()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].info()

for col in binary_cols:
    label_encoder(df,  col)

#One-Hot-Encoding
#######################################################################

df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

def one_hot_encoding(dataframe, cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cols, drop_first=drop_first)
    return dataframe

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoding(df, ohe_cols).head()

#Rare Encoding
#######################################################################

#1.Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / dataframe.shape[0]}))
    print("***************************************************")
    if plot:
        sns.countplot(dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

#2. Seyrek değişkenlerin analizi
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": 100 * dataframe[col].value_counts() / dataframe.shape[0],
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#3. Rare Encoder'in yazılması
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and
                    (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])
    return temp_df

new_df = rare_encoder(df, 0.01)
rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()
new_df["OCCUPATION_TYPE"].value_counts()

##############################################################
#ÖZELLİK ÖLÇEKLENDİRME (FEATURE SCALING)
##############################################################
"""
NEDEN ÖZELLİK ÖLÇEKLENDİRME YAPARIZ?
    1-Değişkenler arasındaki ölçüm farklılığını gidermektir.
    2-Gradient descent kullanan algoritmaların train sürelerini kısaltmak için kullanılır.
    3-Uzaklık temelli algoritmalarda farklı ölçeklerin yanlılığa sebep olmasıdır.
"""

#1.STANDARD SCALER: Klasik standardlaştırma yapılır. Ortalamayı çıkar, standard sapmaya böl. z = (x - u) / s
#########################################################################################################################
df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

#NEDEN df["Age"] değil de df[["Age"]] gönderdik?
#StandardScaler iki boyutlu veri ister df["Age"] Series döndürür, df[["Age"]] ise dataframe döndürür.

df.head()
df["Age_standard_scaler"].min()
df["Age_standard_scaler"].max()

#ROUBAST SCALER: Medyanı çıkar iqr'a böl. Eksik ve aykırı değerlerden etkilenmemek için non-parametrik metrikler kullanılır.
########################################################################################################################################
rs = RobustScaler()
df["Age_roubast_scaler"] = rs.fit_transform(df[["Age"]])

df["Age_roubast_scaler"].min()
df["Age_roubast_scaler"].max()

#MINMAX SCALER: Verilen iki değer arasına ölçekler. Default olarak 0 ile 1 arasına ölçekler.
###########################################################################################################
mms = MinMaxScaler()
df["Age_minmax_scaler"] = mms.fit_transform(df[["Age"]])

df["Age_minmax_scaler"].min()
df["Age_minmax_scaler"].max()

#Karşılaştırma
age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in age_cols:
    num_summary(df, col, plot=True)

#######################################################################
#Sayısal değişkenleri kategorik değişkenlere çevirme
#######################################################################
#qcut fonksiyonu ile bu işlemi yaparız

df["Age_qcut"] = pd.qcut(df["Age"], 5)
df.head()
df.info()

