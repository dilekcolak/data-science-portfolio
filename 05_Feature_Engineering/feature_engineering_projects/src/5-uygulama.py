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
    dataframe = pd.read_csv("datasets/titanic.csv")
    return dataframe

df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

df.head()


#NEW_CABIN_BOOL = cabin null mı değil mi
df["NEW_CABIN_BOOL"] = df["CABIN"].notna().astype("int")

#NEW_NAME_COUNT = name değişkenindeki karakter sayısı
df["NEW_NAME_COUNT"] = df["NAME"].str.len()

#NEW_NAME_WORD_COUNT = split ile boşluklardan kelimeyi ayrıştır sonra kelimeleri say
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

#NEW_NAME_DR = split ile böl dr ile başlayanların lenini al
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split(" ") if x.startswith("Dr")]))

#NEW_TITLE = regex ile NAME'deki unvanları yakalamak istiyoruz. ([A-Za-z]+)\.
df["NEW_TITLE"] = df["NAME"].str.extract("([A-Za-z]+)\.", expand=False)

#NEW_FAMILY_SIZE = sibsp ve parch toplamından ailedeki kişi sayısını bulmak. (kendisini de dahil et)
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

#NEW_AGE_PCLASS = age ve pclass çarpımından yeni değişken türet
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

#NEW_IS_ALONE = sibsp ve parch sayısına göre kişinin tek olup olmadığını belirle. loc kullan
df.loc[((df["SIBSP"] + df["PARCH"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_IS_ALONE"] = "YES"

#NEW_AGE_CAT = 18'den küçükse young, 56'dan küçükse mature, değilse senior olarak tanımla.
df.loc[df["AGE"] <= 18, "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] > 18) & (df["AGE"] <= 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] > 56), "NEW_AGE_CAT"] = "senior"

#NEW_SEX_CAT = yaş grubunu cinsiyete göre özelleştir.
df.loc[(df["NEW_AGE_CAT"] == "young") & (df["SEX"] == "male"), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["NEW_AGE_CAT"] == "mature") & (df["SEX"] == "male"), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["NEW_AGE_CAT"] == "senior") & (df["SEX"] == "male"), "NEW_SEX_CAT"] = "seniormale"

df.loc[(df["NEW_AGE_CAT"] == "young") & (df["SEX"] == "female"), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["NEW_AGE_CAT"] == "mature") & (df["SEX"] == "female"), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["NEW_AGE_CAT"] == "senior") & (df["SEX"] == "female"), "NEW_SEX_CAT"] = "seniorfemale"

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                   and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if col not in num_but_cat
                and dataframe[col].dtypes != "O"]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PASSENGERID"]

def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    lower_bound = quartile1 - 1.5 * iqr
    upper_bound = quartile3 + 1.5 * iqr
    return lower_bound, upper_bound

def check_outliers(dataframe, col_name, q1=0.05, q3=0.95):
    low, up = outlier_threshold(dataframe, col_name, q1, q3)

    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    low, up = outlier_threshold(dataframe, col_name, q1, q3)

    dataframe.loc[(dataframe[col_name] < low), col_name] = low
    dataframe.loc[(dataframe[col_name] > up), col_name] = up

for col in num_cols:
    print(col, check_outliers(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

"""
missing_values_table():
na_columns = null değer içeren sütunları filtrele
n_miss = eksik değer sayısına göre azalan şekilde sırala
ratio = eksik değer sayısının oranını hesapla
missing_df = n_miss ve ratio değerlerini concat ile birleştir 
missing_df'i yazdır
na_name true ise na_columns döndür 
"""

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (100 * dataframe[na_columns].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, axis=1, inplace=True)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
df.loc[df["AGE"] <= 18, "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] > 18) & (df["AGE"] <= 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] > 56), "NEW_AGE_CAT"] = "senior"
df.loc[(df["NEW_AGE_CAT"] == "young") & (df["SEX"] == "male"), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["NEW_AGE_CAT"] == "mature") & (df["SEX"] == "male"), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["NEW_AGE_CAT"] == "senior") & (df["SEX"] == "male"), "NEW_SEX_CAT"] = "seniormale"
df.loc[(df["NEW_AGE_CAT"] == "young") & (df["SEX"] == "female"), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["NEW_AGE_CAT"] == "mature") & (df["SEX"] == "female"), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["NEW_AGE_CAT"] == "senior") & (df["SEX"] == "female"), "NEW_SEX_CAT"] = "seniorfemale"

df.head()

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

missing_values_table(df)

def label_encoder(dataframe, col_name):
    label_encoder = LabelEncoder()
    dataframe[col_name] = label_encoder.fit_transform(dataframe[col_name])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SURVIVED", cat_cols)

"""
rare_encoder(dataframe, rare_perc)
temp_df = dataframe kopyasını oluştur
rare_columns = veri tipi object olan ve frekansının oranı rare_perc'den küçük olan sütunları seç
rare columns içinde for döngüsü = 
    tmp = değişkenin oranı
    rare_labels = sütundaki değerin oranı perc'ten küçük olanların indexi
    temp_df[var] = rare_labels içindeki değerleri where ile Rare yap
temp_df'i döndür
"""

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for col in rare_columns:
        tmp = temp_df[col].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), "Rare", temp_df[col])
    return temp_df

df = rare_encoder(df, 0.01)
rare_analyser(df, "SURVIVED", cat_cols)

def one_hot_encoder(dataframe, cat_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in cat_cols if df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape

#52 tane değişken oldu da bunların hepsi gerekli mi?

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PASSENGERID"]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2
                and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

df.drop(useless_cols, axis=1, inplace=True)

df.head()

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

##########################################
#MODEL
##########################################
y = df["SURVIVED"]
x = df.drop(["SURVIVED", "PASSENGERID"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

accuracy_score(y_pred, y_test)

def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature":features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(rf_model, x_train)