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
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" %x)
pd.set_option("display.width", 500)

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

dff = load_application_train()
dff.head()

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

################################################################
#1. Aykırı Değerleri Yakalama
################################################################

#Grafik Teknikle Aykırı Değerler
####################################
sns.boxplot(x=df["Age"], data=df)
#plt.show()

#Aykırı değerler nasıl hesaplanır?

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)

iqr = q3 - q1

low = q1 - 1.5 * iqr
up = q3 + 1.5 * iqr

df[(df["Age"] > up) | (df["Age"] < low)].index

df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)

def outlier_threshold(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    iqr = quartile3 - quartile1

    low = quartile1 - 1.5 * iqr
    up = quartile3 + 1.5 * iqr

    return low, up

outlier_threshold(df, "Age")

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low, up = outlier_threshold(dataframe, col_name, q1, q3)

    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")


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
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))

#cat_cols, num_cols, cat_but_car = grab_col_names(dff)

#for col in num_cols:
#    print(col, check_outlier(dff, col, 0.01, 0.99))

#Aykırı değerleri belirlemek
###################################
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_threshold(dataframe, col_name)

    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].head())
    else:
        print(dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)])

    if index:
        outlier_index = dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].index
        return outlier_index

grab_outliers(df, "Age")

grab_outliers(df, "Age", True)

#Aykırı değer problemini çözme
#####################################

#1. Aykırı değerleri silebiliriz
low, up = outlier_threshold(df, "Fare")
df.shape

#df = df[~((df["Fare"] > up) | (df["Fare"] < low))]

def remove_outliers(dataframe, col_name):
    low, up = outlier_threshold(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] > up) |(dataframe[col_name] < low))]
    return df_without_outliers

df.shape

for col in num_cols:
    new_df = remove_outliers(df, col)
new_df.shape

#2. Baskılama ile aykırı değerlerden kurtulma
def replace_outliers_with_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    low, up = outlier_threshold(dataframe, col_name, q1, q3)
    dataframe.loc[(dataframe[col_name] > up), col_name] = up
    dataframe.loc[(dataframe[col_name] < low), col_name] = low

for col in num_cols:
    replace_outliers_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#################################################################
#Çok değişkenli aykırı değer analizi: Local Outlier Factor (LOF)
#################################################################
"""
3 kere evlenmiş olmak aykırı değer değildir,
17 yaşında olmak aykırı değer değildir
Peki ya 17 yaşında olup 3 kere evelnmiş olmak aykırı değer midir? 
EVET
"""

"""
Mülakat Sorusu=>
Çok fazla sayıda değişkenimiz var(100 değişken) bunu iki boyutta nasıl görselleştirebiliriz?

bu 100 değişkenin taşıdığı bilginin büyük miktarını taşıdığını varsayabileceğimiz 2 değişkene indirgeyebilirsek 2D görselleştirme yapabiliriz.
PCA (Principle Component Analysis) => Temel Bileşen Analizi ile bu işlemi yapabilriz.
"""

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64", "int64"])
df = df.dropna()
df.head()
df.shape

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_threshold(df, "carat")
df[(df["carat"] < low) | (df["carat"] > up)].shape

low, up = outlier_threshold(df, "depth")
df[(df["depth"] < low) | (df["depth"] > up)].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]

#-1'den -10'a doğru gittikçe değerler aykırılaşır, yalnızlaşır

np.sort(df_scores)[0:5]

#threshold belirlememiz gerekiyor. ELBOW yöntemi kullanacağız
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()
#en son sert değişikliğin yaşandığı noktayı eşik değer olarak seçiyoruz

th = np.sort(df_scores)[3]

df[df_scores < th]

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

#Ağaç yöntemleri kullanacaksak aykırı değerleri ellemeye gerek yokk

##########################################################
#2. Eksik değerlerin yakalanması
##########################################################
df = load()
df.head()

#eksik gözlem var mı yok mu?
df.isnull().values.any()

#değişkenlerdeki eksik değer sayısı
df.isnull().sum()

#değişkenlerdeki tam değer sayısı
df.notnull().sum()

#tüm verideki toplam kaç hücrede eksik değer var?
df.isnull().sum().sum()

#eksik değere sahip satırları gözlemlemek için
df[df.isnull().any(axis=1)]

#eksik değeri olmayan gözlemlere erişmek için
df[df.notnull().all(axis=1)]

#azalan şekilde sırala
df.isnull().sum().sort_values(ascending=False)

100 * df.isnull().sum() / len(df)

#Eksik değere sahip değişkenler
[col for col in df.columns if df[col].isnull().sum() > 0]

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in df.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (100 * dataframe[na_columns].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

#######################################################################
#Eksik değer problemini çözme
#######################################################################

missing_values_table(df)

#ağaç yöntemi kullanacaksak aykırı değerler gibi eksik değerleri de göz ardı edebiliriz

#Çözüm1: Hızlıca silmek
df.dropna().shape #veri boyutu çok az kaldı

#Çözüm2: Basit atama yöntemleri ile doldurmak
df["Age"].fillna(df["Age"].mean()).isnull().sum()

df["Age"].fillna(df["Age"].median()).isnull().sum()

df["Age"].fillna(0).isnull().sum()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
df["Embarked"].fillna("missing").isnull().sum()

dff = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == "O" and len(x.unique()) <= 10 else x, axis=0)


#Çözüm3: Kategorik değer kırılımında değer atama
df.groupby("Sex")["Age"].mean()

df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

#burada aslında ne yapıyoruz
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df["Age"].isnull().sum()

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#Çözüm4: Tahmine dayalı atama ile doldurma
"""
makine öğrenmesi yöntemleri ile dolduracağız (gelişmiş yöntem)
eksiklik olan değişkeni bağımlı deişken olarak alıp diğerlerine bağımsız değişken diyeceğiz
"""

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols  if col not in "PassengerId"]

#One-hot-encoder
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

#değişken standartlaştırma
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

#KNN uygulanması
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

#standartlaştırma işlemini geri alma
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

#atadığımız değerleri görelim
df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]

###########################################################################
#Gelişmiş analizler
###########################################################################

#Eksik verinin incelenmesi
msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

#eksik değerlerin bağımlı değişken ile ilişkisinin incelenmesi
missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Survived", na_cols)

#############################################################################
#3.Encoding (label encoding, One-Hot encoding, Rare encoding)
#############################################################################
df = load()
df.head()

#1.Label Encoding (Binary Encoding - iki değişkenliyse)
#################################################################
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]

le.inverse_transform([0,1])

def label_encoder(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)
#eksik değerlere de 2 değeri atanır. Bunun için öncesinde eksik değer için işlem yapmalıyız veya bu şekilde kabul etmeliyiz.

df[binary_cols].head()

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique()) #NA değerlerini de ayrı bir değişken gibi kabul eder


#2.One-Hot Encoder
#############################################################
df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()
"""
ilk sınıf değeri S, ikinci sınıf değeri C ve üçüncü Q olmasına göre one-hot encoding yapıldı. 
Fakat dummy değişken tuzağına düşmemek için drop_first argümanını True yapıyoruz.
Alfabetik sıraya göre ilk sırada yer alan değişkeni siler.
"""
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

#değişkendeki eksik değerleri de sınıf olarak almak istiyorsak
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

#get_dummies ile hem label encoding hem de one-hot encoding yapabiliriz
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

#cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()

df.head()

#3. Rare Encoding
#######################################################
"""
1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
3. Rare encoder yazılması
"""

#KAtegorik değişkenlerin azlık çokluk durumunun analiz edilmesi
df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

#Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyzer(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyzer(df, "TARGET", cat_cols)

#Rare Encoder fonksiyonunun yazılması
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyzer(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()

##########################################################
#Feature Scaling (Özellik Ölçeklendirme)
##########################################################

#1. STANDARD SCALER:
###########################################################
"""
 Klasik standartlaştırma. 
 Ortalamayı çıkar,
 Standart sapmaya böl.
 z = (x - u) / s
 
!!!!AYKIRI DEĞERLERDEN ETKİLENİR!!!!!!
"""
df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

#2.ROBUST SCALER
"""
Medyanı çıkar
iqr'a böl
!!!AYKIRI DEĞERLERE DAYANIKLIDIR!!!!!
Ama çok yaygın bir kullanımı yoktur.
"""
rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

#3.MinMaxScaler
"""
Verilen iki değer arasında değişken dönüşümü.
default olarak 0 ile 1 arasına ölçekler.
x_std = (x - x_min) / (x_max - x_min)
x_scaled = x_std * (max - min) + min
"""
mms = MinMaxScaler()
df["Age_min_mix_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col,plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title("Histogram of " + numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)


######################################################
#Numeric to Categoric (Sayısal değişkenleri kategorik değişkenlere çevirmek
#BINNING
######################################################
df["Age_qcut"] = pd.qcut(df["Age"], 5)
df.head()


#######################################################
#Feature Extraction (Özellik çıkarımı)
#######################################################

#Binary Features : Flag, Bool, True-False
###############################################

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])
print(f"Test Stat = {test_stat}, p-value = {pvalue}")

#p-value < 0.05 olduğu için H0 hipotezi reddedilir.
#Yani cabin bilgisi olup olmaması hayatta kalma ihtimalini etkiliyor.

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.head()

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])
print(f"Test Stat = {test_stat}, p-value = {pvalue}")

#H0 reddedilir. Hayatta kalma oranını bu yeni değişken de etkiler.

#Text'ler üzerinden özellik üretmek
#############################################################
df = load()
df.head()

#letter count
df["NEW_NAME_COUNT"] = df["Name"].str.len()

#word count
df["NEW_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

#Özel Yapıları Yakalamak
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

#Regex ile değişken türetmek
df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean",
                                                                 "Age": ["count", "mean"]})

########################################################
#Date değişkenleri türetmek
########################################################
dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

#tarih değişkeninin türünü date yapalım (object olarak kullanamayız)
dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d %H:%M:%S")

#year
dff["year"] = dff["Timestamp"].dt.year

#month
dff["month"] = dff["Timestamp"].dt.month

#year difference
dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year

#month difference (iki taarh arasındaki ay farkı): yıl farkı + ay farkı
dff["month_diff"] = (date.today().year- dff["Timestamp"].dt.year) * 12 + date.today().month - dff["Timestamp"].dt.month

#day name
dff["day_name"] = dff["Timestamp"].dt.day_name()

dff.head()

#######################################################################
#Feature Interactions (Özellik etkleşimleri)
#######################################################################
df = load()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "male") & ((df["Age"] > 21) & (df["Age"]) <= 50), "NEW_SEX_CAT"] = "maturemale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniormale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"

df.loc[(df["Sex"] == "female") & ((df["Age"] > 21) & (df["Age"]) <= 50), "NEW_SEX_CAT"] = "maturefemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()