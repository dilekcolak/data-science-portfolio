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

dff = load_application_train()
dff.head()

df = load()
df.head()

#######################################################################
#Aykırı değerleri yakalamak
#######################################################################

#Grafik teknik ile yakalamak
#######################################################################
sns.boxplot(x=df["Age"])
plt.show()


#######################################################################
#Aykırı değerler nasıl yakalanır?
#######################################################################
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)

iqr = q3 - q1

up_limit = q3 + 1.5 * iqr
low_limit = q1 - 1.5 * iqr

df[(df["Age"] > up_limit) | (df["Age"] < low_limit)]

df[(df["Age"] > up_limit) | (df["Age"] < low_limit)].index

df[(df["Age"] > up_limit) | (df["Age"] < low_limit)].any(axis=None)

def outlier_thresholds(dataframe, value, q1=0.01, q3=0.99):
    quartile1 = dataframe[value].quantile(q1)
    quartile3 = dataframe[value].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

outlier_thresholds(df, "Age")

def check_outlier(dataframe, value, q1=0.01, q3=0.99, index=False):
    low, up = outlier_thresholds(dataframe, value, q1, q3)
    if dataframe[(dataframe[value] > up) | (dataframe[value] < low)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age", 0.25, 0.75)

###############################################################
#grab_col_names
###############################################################

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))

cat_cols, num_cols, cat_but_car = grab_col_names(dff)
num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]
for col in num_cols:
    print(col, check_outlier(dff, col))

#############################################################################
#Aykırı değerlere arişmek
#############################################################################

def grab_outliers(dataframe, col_name, index=False, q1=0.01, q3=0.99):
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)

    if dataframe[((dataframe[col_name] > up) | (dataframe[col_name] < low))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] > up) | (dataframe[col_name] < low))].head())
    else:
        print(dataframe[((dataframe[col_name] > up) | (dataframe[col_name] < low))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] > up) | (dataframe[col_name] < low))].index
        return outlier_index

check_outlier(df, "Age", 0.25, 0.75)
grab_outliers(df, "Age", q1=0.25, q3=0.75)
grab_outliers(df, "Age", q1=0.25, q3=0.75, index=True)

#############################################################################
#Aykırı değer problemini çözme
#############################################################################

#1)Silerek aykırı değerlerden kurtulabiliriz
#################################################
def remove_outlier(dataframe, col_name, q1=0.01, q3=0.99):
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low) | (dataframe[col_name] > up))]
    return df_without_outliers

check_outlier(df, "Age", 0.25, 0.75)
df.shape
df = remove_outlier(df, "Age", q1=0.25, q3=0.75)
check_outlier(df, "Age", 0.25, 0.75)
df.shape

low, up = outlier_thresholds(df, "Fare", q1=0.25, q3=0.75)
df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]
df.loc[((df["Fare"] < low) |(df["Fare"] > up)), "Fare"]

df.loc[df["Fare"] > up, "Fare"] = up
check_outlier(df, "Fare", q1=0.25, q3=0.75)

def replace_with_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)
    df.loc[dataframe[col_name] < low, col_name] = low
    df.loc[dataframe[col_name] > up, col_name] = up

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col, 0.25, 0.75))
    if check_outlier(df, col, 0.25, 0.75):
        replace_with_thresholds(df, col, q1=0.25, q3=0.75)
        print(col, check_outlier(df, col, 0.25, 0.75))


#############################################
#Recap
#############################################
df = load()
outlier_thresholds(df, "Age", q1=0.25, q3=0.75)
check_outlier(df, "Age", 0.25, 0.75)
grab_outliers(df, "Age", q1=0.25, q3=0.75)
grab_outliers(df, "Age", q1=0.25, q3=0.75, index=True)

df.shape
remove_outlier(df, "Age", q1=0.25, q3=0.75).shape
replace_with_thresholds(df, "Age", q1=0.25, q3=0.75)
check_outlier(df, "Age", 0.25, 0.75)


###########################################################################
#Çok değişkenli aykırı değer analizi(Local Outliar Factor -> LOF)
###########################################################################
df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64", "int64"])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col, 0.25, 0.75))

low, up = outlier_thresholds(df, "carat", q1=0.25, q3=0.75)
df[((df["carat"] < low) | (df["carat"] > up))].shape


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_

df_scores[0:5]
np.sort(df_scores)[0:5]

################################################
#Elbow yöntemi ile eşik değer belirleme
#########################################################
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style=".-")
plt.show()

#En son marjinal değişikliğin yaşandığı yeri threshold olarak kabul ediyoruz.

th = np.sort(df_scores)[3]

df[df_scores < th]
df[df_scores < th].shape

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)
