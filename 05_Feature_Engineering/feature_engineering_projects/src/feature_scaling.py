"""
FEATURE SCALING:
    Değişkenler arasındaki ölçüm farkını gidermek isteriz.
    Tüm değişkenleri eşit şartlar altında değerlendirmeliyiz.
    Mesela yapılan yorum sayısı ile ratingin etkisi aynı olmalı
    ama yorum sayısı değer bazında çook büyük olduğu için etkisi
    daha baskın olur.

    Gradient descent gibi iteratif çalışan algoritmalar için
    sonuca ulaşmayı daha tutarlı hale getirir ve hesaplama süresini kısaltır.

    Standard Scaler: standard sapma ve ortalama ile işlem yaptığı için aykırı değerlerden etkilenir.
                     ortalamayı 0, standard sapmayı 1 yapar.
    Robust Scaler: medyan ve iqr kullanarak aykırı değer etkilerinden kaçınır.
                   medyanı 0 yapar.
    MinMaxScaler: Değerleri dönüştürmek istediğimiz aralığa getirir.
                  default olarak 0 ile 1 arasına ölçekler.
    Sayısal değişkenleri kategorik değişkenlere dönüştürmek için qcut veya cut fonksiyonunu kullanabiliriz.
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
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" %x)
pd.set_option("display.width", 500)

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

#StandardScaler
df = load()
ss= StandardScaler()
df.head()

df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

#RobustScaler
rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

#MinMaxScaler
mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]

df[age_cols].head(10)

#Numeric to Categorical (Binning)
df["age_qcut"] = pd.qcut(df["Age"], 5)


