import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import missingno as msno
from datetime import date

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

"""
Adım 1: Genel resmi inceleyiniz.
Adım 2: Numerikvekategorikdeğişkenleri yakalayınız.
Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
Adım 4:Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre 
numerik değişkenlerin ortalaması)
Adım 5: Aykırı gözlem analizi yapınız.
Adım 6: Eksik gözlem analizi yapınız.
Adım 7: Korelasyon analizi yapınız
"""
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" %x)
pd.set_option("display.width", 500)
df = pd.read_csv("datasets/diabetes.csv")
target = "Outcome"

print(df.head())
print(df.info())
print(df.describe().T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype != "O"
                   and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                and col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print(f"cat_but_car: {len(cat_but_car)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def target_summary_with_num_cols(dataframe, target, num_col):
    print(dataframe.groupby(target)[num_col].mean(), end="\n\n\n")

for col in num_cols:
    target_summary_with_num_cols(df, target, col)

def cat_summary_with_target(dataframe, target, cat_col):
    print(dataframe.groupby(cat_col)[target].mean(), end="\n\n\n")

for col in cat_cols:
    cat_summary_with_target(df, target, col)

def outlier_threshold(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    iqr = quartile3 - quartile1

    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_threshold(dataframe, col_name, q1, q3)

    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_threshold(dataframe, col_name, q1, q3)
    outliers = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]
    return outliers

def outlier_summary(dataframe, col_name, q1=0.25, q3=0.75):
    outliers = grab_outliers(dataframe, col_name, q1, q3)

    print(f"################ Outliers Summary for {col_name} #############")
    print("Outlier Counts:", outliers.shape[0])
    print("Outlier Ratio:", 100 * outliers.shape[0] / dataframe.shape[0], end="\n\n\n")

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

for col in num_cols:
    if check_outlier(df, col, q1=0.05, q3=0.95):
        outlier_summary(df, col, q1=0.05, q3=0.95)

#Eksik değerleri gözlemlemek için kullanabileceğimiz yöntemler
msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

def missing_values(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (100 * dataframe[na_columns].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values(df)

def plot_correlation_heatmap(dataframe, figsize=(10,8)):
    corr = dataframe.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="RdBu", fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix")
    plt.show()

plot_correlation_heatmap(df)


###########
#Base Model Kurulumu
###########

Y = df["Outcome"]
X = df.drop("Outcome", axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=42).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 4)}")
print(f"Recall: {round(recall_score(y_test, y_pred), 4)}")
print(f"Precision {round(precision_score(y_test, y_pred), 4)}")
print(f"F1: {round(f1_score(y_test, y_pred), 4)}")
print(f"Roc-Auc: {round(roc_auc_score(y_test, y_pred), 4)}")

"""
RESULTS:
Accuracy: 0.7619
Recall: 0.5679
Precision 0.697
F1: 0.6259
Roc-Auc: 0.7173
"""

"""
Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. 
değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0 
olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik 
değerlere işlemleri uygulayabilirsiniz.
Adım 2: Yeni değişkenler oluşturunuz.
Adım 3: Encoding işlemlerini gerçekleştiriniz.
Adım 4: Numerik değişkenler için standartlaştırma yapınız.
Adım 5: Model oluşturunuz
"""

def plot_distribution(dataframe, col_name, bins=30):
    plt.figure(figsize=(8, 5))
    sns.histplot(dataframe[col_name], bins=bins, kde=True)
    plt.title(f"Distribution of {col_name}")
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.show(block=True)

for col in num_cols:
    plot_distribution(df, col)

#Bazı değişkenlere nan yerine 0 değerleri atanmış bunları nan yapmalıyız.
#Örneğin kan basıncı 0 olabilir mi? hayır bu insan ölü olur.

#Glucose, BloodPressure, SkinThickness, Insulin, BMI => değiştirilmesi gereken sütunlar

def zero_to_nan(dataframe, cols):
    for col in cols:
        dataframe[col] = dataframe[col].replace(0, np.nan)
    return dataframe

zero_as_nan_cols = ["BMI", "BloodPressure", "SkinThickness", "Insulin", "Glucose"]

df = zero_to_nan(df, zero_as_nan_cols)

df.isnull().sum()

msno.matrix(df)
plt.show()

missing_values(df)

def impute_with_median(dataframe, selected_col):
    dataframe[selected_col] = dataframe[selected_col].fillna(dataframe[selected_col].median())
    return dataframe

def selected_correlated_cols(dataframe, selected_col, target=None, th=0.25):
    corr = dataframe.corr(numeric_only=True)[selected_col].sort_values(ascending=False)

    exclude = [selected_col]
    if target is not None:
        exclude.append(target)

    correlated_cols = [col for col, val in corr.items()
                       if col not in exclude and
                       abs(pd.notnull(val)) >= th][:2]
    return correlated_cols

#korelasyonu yüksek olan sütunlara göre değer atama
def hybrid_impute(dataframe, selected_col, target=None, th=0.25):
    feats = selected_correlated_cols(dataframe, selected_col, target, th)

    print(selected_col, " ilk na sayısı: ", dataframe[selected_col].isnull().sum())

    if len(feats) < 2:
        return impute_with_median(dataframe, selected_col)

    train_df = dataframe[dataframe[selected_col].notnull()][feats + [selected_col]].dropna()
    test_df = dataframe[dataframe[selected_col].isnull()][feats].dropna()

    if train_df.shape[0] == 0 or test_df.shape[0] == 0:
        print(selected_col, "train_df/test_df boş. median ile devam ediliyor.")
        return impute_with_median(dataframe, selected_col)

    x_train= train_df[feats]
    y_train= train_df[selected_col]

    model = LinearRegression().fit(x_train, y_train)

    dataframe.loc[test_df.index, selected_col] = model.predict(test_df)

    print(selected_col, "için regresyon sonrası na sayısı:", dataframe[selected_col].isnull().sum())

    #yine de nan değer kaldıysa diye
    if dataframe[selected_col].isnull().sum() > 0:
        dataframe = impute_with_median(dataframe, selected_col)

    dataframe[selected_col] = dataframe[selected_col].clip(lower=0)

    print(selected_col, "için son durumda na sayısı: ", dataframe[selected_col].isnull().sum(), end="\n\n")
    return dataframe


zero_as_nan_cols = ["BMI", "BloodPressure", "SkinThickness", "Insulin", "Glucose"]

used = {}
for col in zero_as_nan_cols:
    feats = selected_correlated_cols(df, selected_col=col, target="Outcome", th=0.25)
    used[col] = feats
    df = hybrid_impute(df, selected_col=col, target="Outcome", th=0.25)

for key, val in used.items():
    print(key, ":",val)

def replace_outliers_with_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_threshold(dataframe, col_name, q1, q3)

    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

for col in num_cols:
    if check_outlier(df, col, q1=0.05, q3=0.95):
        outlier_summary(df, col, q1=0.05, q3=0.95)
    else:
        print(col, "için aykırı değer bulunmamaktadır.", end="\n\n")

for col in num_cols:
    if check_outlier(df, col, q1=0.05, q3=0.95):
        replace_outliers_with_thresholds(df, col, q1=0.05, q3=0.95)

bmi_bins = [0, 18.5, 24.9, 29.9, float("inf")]
labels = ["Underweight", "Healthy", "Overweight", "Obese"]

df["NEW_BMI_CATEGORY"] = pd.cut(df["BMI"], bins=bmi_bins, labels=labels)

df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[df["Age"] >= 50, "NEW_AGE_CAT"] = "senior"

df["NEW_GLUCOSE"] = pd.cut(df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

df.loc[((df["Age"] >= 21) & (df["Age"] < 50)) & (df["NEW_BMI_CATEGORY"] == "Underweight"), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["Age"] >= 50) & (df["NEW_BMI_CATEGORY"] == "Underweight"), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["Age"] >= 21) & (df["Age"] < 50)) & (df["NEW_BMI_CATEGORY"] == "Healthy"), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[(df["Age"] >= 50) & (df["NEW_BMI_CATEGORY"] == "Healthy"), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["Age"] >= 21) & (df["Age"] < 50)) & (df["NEW_BMI_CATEGORY"] == "Overweight"), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[(df["Age"] >= 50) & (df["NEW_BMI_CATEGORY"] == "Overweight"), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[((df["Age"] >= 21) & (df["Age"] < 50)) & (df["NEW_BMI_CATEGORY"] == "Obese"), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["Age"] >= 50) & (df["NEW_BMI_CATEGORY"] == "Obese"), "NEW_AGE_BMI_NOM"] = "obesesenior"


df.loc[((df["Age"] >= 21) & (df["Age"] < 50)) & (df["Glucose"] < 70), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Age"] >= 50) & (df["Glucose"] < 70), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Age"] >= 21) & (df["Age"] < 50)) & ((df["Glucose"] >= 70) & (df["Glucose"] < 100)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[(df["Age"] >= 50) & ((df["Glucose"] >= 70) & (df["Glucose"] < 100)), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Age"] >= 21) & (df["Age"] < 50)) & ((df["Glucose"] >= 100) & (df["Glucose"] < 125)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[(df["Age"] >= 50) & ((df["Glucose"] >= 100) & (df["Glucose"] < 125)), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[((df["Age"] >= 21) & (df["Age"] < 50)) & (df["Glucose"] >= 125), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Age"] >= 50) & (df["Glucose"] >= 125), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]
df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

df.columns = [col.upper() for col in df.columns]
df.head()

df.loc[df["NEW_AGE_BMI_NOM"].isnull(), ["Age", "BMI", "NEW_AGE_BMI_NOM"]]
df["NEW_AGE_BMI_NOM"].unique()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.shape
df.head()

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]

def one_not_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_not_encoder(df, cat_cols, drop_first=True)

df.shape
df.head()

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.shape
df.head()
df.describe().T

####################
#MODELLEME
####################

Y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=42).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 4)}")
print(f"Recall: {round(recall_score(y_test, y_pred), 4)}")
print(f"Precision {round(precision_score(y_test, y_pred), 4)}")
print(f"F1: {round(f1_score(y_test, y_pred), 4)}")
print(f"Roc-Auc: {round(roc_auc_score(y_test, y_pred), 4)}")

"""
RESULTS:
Accuracy: 0.7879
Recall: 0.6173
Precision 0.7353
F1: 0.6711
Roc-Auc: 0.7486
"""

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_,
                                "Feature": features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(rf_model, X)
