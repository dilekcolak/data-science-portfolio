"""
Telco Churn Feature Engineering
İş Problemi:
    Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
    geliştirilmesi istenmektedir.  Modeli geliştirmeden önce gerekli olan veri analizi
    ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

Veri Seti Hikayesi:
    Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
    bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu
    gösterir.

Görev 1 : Keşifçi Veri Analizi
    Adım 1: Genel resmi inceleyiniz.
    Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
    Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
    Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
    Adım 5: Aykırı gözlem analizi yapınız.
    Adım 6: Eksik gözlem analizi yapınız.
    Adım 7: Korelasyon analizi yapınız.
Görev 2 : Feature Engineering
    Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
    Adım 2: Yeni değişkenler oluşturunuz.
    Adım 3: Encoding işlemlerini gerçekleştiriniz.
    Adım 4: Numerik değişkenler için standartlaştırma yapınız.
    Adım 5: Model oluşturunuz.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option("display.float_format", lambda x: "%.2f" %x)
pd.set_option("display.width", 500)

def load():
    data = pd.read_csv("datasets/Telco-Customer-Churn.csv")
    return data

df = load()

#Görev 1
df.head()
df.shape
df.info()

#TotalCharges ve Churn değişkenleri sayısal olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].str.strip().str.lower().apply(lambda x: 1 if x == "yes" else 0)

df["Churn"].value_counts()
df.isnull().sum()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                   and dataframe[col].nunique() < cat_th]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                and col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorical Columns: {cat_cols}")
    print(f"Numerical Columns: {num_cols}")
    print(f"Categoric but Cardinal Columns: {cat_but_car}")
    print(f"Numeric but Categorical Columns: {num_but_cat}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

def num_col_analysis(dataframe, num_cols, target):
    for col in num_cols:
        print(dataframe.groupby(target).agg({col: "mean"}), end="\n\n")

num_col_analysis(df, num_cols, "Churn")

def cat_col_analysis(dataframe, cat_cols, target):
    for col in cat_cols:
        print(dataframe.groupby(col).agg({target: "mean"}), end="\n\n")

cat_cols = [col for col in cat_cols if col != "Churn"]
cat_col_analysis(df, cat_cols, "Churn")

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

def outlier_thresholds(dataframe, col, q1=0.01, q3=0.99):
    quartile1 = dataframe[col].quantile(q1)
    quartile3 = dataframe[col].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit

def check_outlier(dataframe, col, q1=0.01, q3=0.99):
    low, up = outlier_thresholds(dataframe, col, q1, q3)
    if df.loc[(dataframe[col] > up) | (dataframe[col] < low), col].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if check_outlier(df, col):
        print(f"{col} has outliers!", end="\n")
    else:
        print(f"{col} has not outliers!", end="\n")

for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"{col} has missing values!", end="\n")
    else:
        print(f"{col} has no missing values!", end="\n")

df[num_cols].corr()
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Heatmap", fontsize=20)
plt.show()

#Görev 2
def replace_with_thresholds(dataframe, col, q1=0.01, q3=0.99):
    low, up = outlier_thresholds(dataframe, col, q1, q3)
    dataframe.loc[(dataframe[col] > up), col] = up
    dataframe.loc[(dataframe[col] < low), col] = low

for col in num_cols:
    replace_with_thresholds(df, col)

#df.dropna(subset=num_cols, inplace=True)
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#BASE MODEL KURULUMU
dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]

def one_hot_encoder(dataframe, cat_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, cat_cols)

dff.shape
dff.head()
dff.dtypes

ss = StandardScaler()
dff[num_cols] = ss.fit_transform(dff[num_cols])
dff[num_cols].describe().T


y = dff["Churn"]
x = dff.drop(["Churn", "customerID"], axis=1)

x.head()
y.head()

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=17)
X_train.shape
Y_train.shape
X_test.shape
Y_test.shape

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, Y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, Y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, Y_test), 4)}")
print(f"Precision: {round(precision_score(y_pred, Y_test), 4)}")
print(f"F1: {round(f1_score(y_pred, Y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred, Y_test), 4)}")

dff.head()

df["NEW_MonthlyCharges_Level"] = pd.qcut(df["MonthlyCharges"], 3, labels=["low", "medium", "high"])
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

df.head()

df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

df["NEW_NoProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["SeniorCitizen"] == 0) and (x["NEW_Engaged"] == 0) else 0, axis=1)

df["NEW_TotalServices"] = (df[["PhoneService", "InternetService", "OnlineSecurity",
                              "OnlineBackup", "DeviceProtection", "TechSupport",
                              "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)

df["NEW_Flag_Any_Streaming"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

df["NEW_Flag_AutoPayment"] = df.apply(lambda x: 1 if ("(automatic)" in x["PaymentMethod"]) else 0, axis=1)

df["NEW_Avg_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

df["NEW_Increase"] = df["NEW_Avg_Charges"] / df["MonthlyCharges"]

df["NEW_Avg_Service_Fee"] = df["MonthlyCharges"] / (df["NEW_TotalServices"] + 1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols = [col for col in cat_cols if col not in ["Churn"]]

def label_encoder(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()
df.shape

cat_cols = [col for col in df.columns if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]

df = one_hot_encoder(df, cat_cols)

#MODELLEME

y = df["Churn"]
x = df.drop(["Churn", "customerID"], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=17)
cat_boost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(x_train, y_train)
y_pred = cat_boost_model.predict(x_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)}")
print(f"F1: {round(f1_score(y_pred, y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 4)}")

def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    plt.figure(figsize=(15, 10))
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    plt.title(model_type + " Feature Importance")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")
    plt.show()

plot_feature_importance(cat_boost_model.get_feature_importance(), x.columns, "CatBoost")

