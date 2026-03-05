"""
encoding: Değişkenlerin temsil şekillerinde değişiklikler yapmaktır

LABEL ENCODING:
    mesela sex değişkeninini is_female değişkenine dönüştürüp 0-1 ile ifade etmektir. (BINARY ENCODING)
    Ordinal kategorik bir değişkeni de sayılar ile ifade edbiliriz. Mesela education değişkeni var ve değerleri:
        "pre_school", "secondary school", "high school", "graduate", "master", "phD"
    Bu değişkenleri 1, 2, 3, 4, 5 diye encode edebiliriz.
    Sıralı olduğu için bunu yapabiliriz.
    Ama mesela futbol takımları için bir değişkenimiz var ve değerleri:
        "galatasaray", "fenerbahçe", "beşiktaş", "tranzonspor"
    Takımmlar arasında herhangi bir sıralama olmadığı için 1, 2, 3, 4 diye sayısal değer atamak yanlış olur.
    neye göre vereceğiz sayıyı?
    LabelEncoder objesine direkt bu değişkeni erseydik alfabetik sırlaamaya göre değer ataması yapardı.
    Modeller bu sayısal değerleri büyüklük olarak algılar ve sonuç üretirken hataya sebep olur.

ONE-HOT ENCODING:
    Takım değişkenimiz ve aldığı 4 farklı değer var ("gs", "fb", "bjk", "ts"). Bu değerlerin her biini bir değişkene dönüştürürüz.
    Yani gs, fb gibi sütunlar oluşturulur ve hangi takıma aitse o 1, düğerleri 0 yapılır.
    Yeni değişkenler oluşturulurken drop_first yaparak ilk değişken uçurulur. Neden?
        Dummy değişken tuzağından kurtulmak için yapılır.

    Dummy değişken tuzağı nedir?
        sütunların birbiri üzerinden oluşturulmasıdır. İki değişken arasında yüksek korelasyon varsa bu ihtimal artar.
    Sınıfların sıralanmasını alfabetik olarak yapar.
    dummy_na parametresi ile NaN değerler için de bir sütun oluşturabiliriz.

    Label encoder kullanmadan one-hot encoder ile binary encoding de yapabiliriz. Sex değişkeni için get_dummies metoduna drop_first argümanı ile
    bu sütunu verirsek binary encoding yapılmış olur.

RARE ENCODING:
    Kategorik bir değişkenin bazı değerlerinin frekansı çok yüksekken bazılarınınki çok düşük olur.
    Frekansı düşük olan değerlerden gereksiz yeni değişken üretmek performansı çok etkiler.
    Bu yüzden rare encoding ile üretilecek değişken sayısını azaltabiliriz.
    Düşük frekanslı değerler birleştirilip bunlardan tek bir değişken oluşturulabilir.
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
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                   and dataframe[col].nunique() < cat_th]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                and col not in num_but_cat]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]

    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

#####################################################
#Label Encoder
#####################################################

df = load()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[:10]

#hangi değere hangi sayının atandığını öğrenmek için inverse_transform kullanabiliriz.
le.inverse_transform([0,1])

def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

binary_cols = [col for col in df.columns if df[col].dtype not in [float, int]
               and df[col].nunique() == 2]

#Neden burada nunique kullandık da len(unique) kullanmadık?
#Çünkü eğer bu değişkende null olan herhangi bir gözlem varsa len(unique) onu da yeni bir değer olarak
#kabul eder. Yani "Sex" değişkeninin kaç kategorisi var normalde 2, ama null değer varsa bize 3 sonucunu döndürür.
df["Embarked"].nunique() #3
len(df["Embarked"].unique()) #4

for col in binary_cols:
    label_encoder(df, col)

df.head(10)


dff = load_application_train()

binary_cols = [col for col in dff.columns if dff[col].dtype not in [float, int]
               and dff[col].nunique() == 2]

for col in binary_cols:
    label_encoder(dff, col)

dff[['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EMERGENCYSTATE_MODE']].head()

#Null değerlere de 2 değeri verilir. Bunun farkında olmamız lazım.

###################################################################
#One-Hot Encoding
###################################################################

df = load()

df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

pd.get_dummies(df, columns=["Sex"], drop_first=True).head()

pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in cat_cols if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()

###################################################################
#Rare Encoding
###################################################################
dff = load_application_train()

dff["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(dff, col)

dff["NAME_INCOME_TYPE"].value_counts()

dff.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": 100 * dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(dff, "TARGET", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df
