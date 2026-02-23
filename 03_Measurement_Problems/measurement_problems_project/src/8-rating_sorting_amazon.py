"""
Rating Product & Sorting Reviews in Amazon
İş Problemi:
    E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır. Bu
    problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
    alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde
    sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem
    maddi kayıp hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını
    arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" %x)

df_ = pd.read_csv("datasets/amazon_review.csv")

df = df_.copy()
df.head()

"""
Görev 1:  Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Bu görevde amacımız verilen puanları tarihe göre 
ağırlıklandırarak değerlendirmek. İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.
    Adım 1:   Ürünün ortalama puanını hesaplayınız.
    Adım 2:  Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
    Adım 3:  Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
"""
df["overall"].mean()
df["day_diff_segment"] = pd.qcut(df["day_diff"], 4, labels=[1, 2, 3, 4])

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff_segment"] == 1, "overall"].mean() * w1 / 100 + \
        dataframe.loc[dataframe["day_diff_segment"] == 2, "overall"].mean() * w2 / 100 + \
        dataframe.loc[dataframe["day_diff_segment"] == 3, "overall"].mean() * w3 / 100 + \
        dataframe.loc[dataframe["day_diff_segment"] == 4, "overall"].mean() * w4 / 100

time_based_weighted_average(df)

df.loc[df["day_diff_segment"] == 1, "overall"].mean()
df.loc[df["day_diff_segment"] == 2, "overall"].mean()
df.loc[df["day_diff_segment"] == 3, "overall"].mean()
df.loc[df["day_diff_segment"] == 4, "overall"].mean()

"""
Görev 2:  Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
    Adım 1:  helpful_no değişkenini üretiniz.
        • total_vote bir yoruma verilen toplam up-downsayısıdır.
        • up, helpful demektir.
        • Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
        • Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.
    Adım 2:  score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.
        • score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff, 
        score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
        • score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisindescore_pos_neg_diffismiyle kaydediniz.
        • score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
        • wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.
    Adım 3:  20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
•       • wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.
•       • Sonuçları yorumlayınız
"""

def pos_neg_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if (up + down) == 0:
        return 0
    else:
        return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df["score_pos_neg_diff"] = df.apply(lambda x: pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values(by="wilson_lower_bound", ascending=False).head(20)