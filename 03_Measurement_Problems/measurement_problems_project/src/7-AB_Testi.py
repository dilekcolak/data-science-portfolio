"""
AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması

İş Problemi:
    Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif olarak yeni bir teklif türü olan
    "averagebidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test
    etmeye karar verdi ve averagebidding'inmaximumbidding'den daha fazla dönüşüm getirip getirmediğinianlamak için birA/B
    testi yapmak istiyor.

    A/B testi 1 aydır devam ediyor ve bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.
    Bombabomba.comiçin nihai başarı ölçütüPurchase'dır. Bu nedenle, istatistiksel testler için Purchase metriğine
    odaklanılmalıdır.
"""
import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" %x)

"""
Görev 1:  Veriyi Hazırlama ve Analiz Etme
    Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
    Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
    Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz. 
"""
control_df = pd.read_excel("datasets/ab_testing.xlsx",
                           sheet_name="Control Group")

test_df = pd.read_excel("datasets/ab_testing.xlsx",
                        sheet_name="Test Group")

control_df.head()
test_df.head()

control_df.describe().T
test_df.describe().T

control_df["group"] = "control"
test_df["group"] = "test"

df = pd.concat([control_df, test_df], ignore_index=True)

df.describe().T
df.head()
df.tail()

"""
Görev 2:  A/B Testinin Hipotezinin Tanımlanması
    Adım 1: Hipotezi tanımlayınız. 
    Adım 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarını analiz ediniz.
"""
#H0: M1 = M2
#H1: M1 != M2

df.groupby("group").agg({"Purchase": "mean"})

"""
Görev 3:  Hipotez Testinin Gerçekleştirilmesi
    Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız. Bunlar Normallik Varsayımı ve VaryansHomojenliğidir. Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz
        Normallik Varsayımı :
            H0: Normal dağılım varsayımı sağlanmaktadır. 
            H1: Normal dağılım varsayımı sağlanmamaktadır.
            p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
            Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini yorumlayınız. 
        VaryansHomojenliği :
            H0: Varyanslar homojendir.
            H1: Varyanslar homojen Değildir.
            p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
            Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz. 
            Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.  
    Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.
    Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.
"""

#Normallik varsayımı:
test_stats, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
test_stats, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])

if pvalue < 0.05:
    print("Normallik varsayımı hipotezi reddedilir.")
else:
    print("Normallik varsayımı hipotezi reddedilemez.")

#Varyans homojenliği varsayımı:
test_stats, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                            df.loc[df["group"] == "test", "Purchase"])
if pvalue < 0.05:
    print("Varyans homojenliği varsayımı hipotezi reddedilir.")
else:
    print("Varyans homojenliği varsayımı hipotezi reddedilemez.")

#Varsayımlar reddedilemediği için parametrik test olan bağımsız iki örneklem t-testi uygularız.
test_stats, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                               df.loc[df["group"] == "test", "Purchase"],
                               equal_var=True)
if pvalue < 0.05:
    print("H0 hipotezi reddedilir.")
    print("Kontrol ve Test grubu satın alma ortalamaları arasında istatistiksel olaraak anlamlı bir farklılık vardır.")
else:
    print("H0 hipotezi reddedilemez.")
    print("Kontrol ve Test grubu satın alma ortalamaları arasında istatistiksel olaraak anlamlı bir farklılık yoktur.")

