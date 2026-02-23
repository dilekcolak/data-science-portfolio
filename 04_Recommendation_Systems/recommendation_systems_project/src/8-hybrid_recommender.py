import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

#USER BASED RECOMMENDATION

########################################################################
#GÖREV 1: Veri Hazırlama
########################################################################

#Adım 1: movie, rating veri setlerini okutunuz.
movie = pd.read_csv("datasets/movie.csv")
rating = pd.read_csv("datasets/rating.csv")

movie.columns
rating.columns

#Adım 2: rating veri setine Id'lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
df = rating.merge(movie, how="left", on="movieId")

df.columns
df.head()
df["title"].nunique()

#Adım 3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkarınız.
rating_counts =  pd.DataFrame(df["title"].value_counts())

rating_counts.shape
rating_counts.head()

rare_movies = rating_counts[rating_counts["count"] < 1000]
rare_movies.shape
rare_movies.index.nunique()
rare_movies.head()

common_movies = df[~df["title"].isin(rare_movies.index)]
common_movies.shape
common_movies["title"].nunique()
common_movies["title"].value_counts()
common_movies.head()

#Adım 4: index'te userID'lerin sütunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz.
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.head()

#Adım 5: Yapılan tüm işlemleri fonksiyonlaştırınız.
def create_rating_table(movie, rating):
    df = rating.merge(movie, how="left", on="movieId")

    rating_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = rating_counts[rating_counts["count"] < 1000]
    common_movies = df[~df["title"].isin(rare_movies.index)]

    rating_table = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return rating_table

um_df = create_rating_table(movie, rating)

um_df.head()

########################################################################
#GÖREV 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Listelenmesi
########################################################################

#Adım 1: Rastgele bir kullanıcı İd seçiniz.
random_user = pd.Series(user_movie_df.index).sample(n=1).iloc[0]

#Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşsan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user]

#Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

len(movies_watched)

########################################################################
#GÖREV 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
########################################################################

#Adım 1:Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturunuz.
movies_watched_df = user_movie_df[movies_watched]

#Adım 2: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "count"]

#Adım 3: Seçilen kullanıcının oy verdiği filmlerin %60 ve üstünü izleyenlerin id'lerinden user_same_movies adında bir liste oluşturunuz.
count = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["count"] >= count]["userId"].tolist()

type(users_same_movies)
len(users_same_movies)

########################################################################
#GÖREV 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
########################################################################

#Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz
movies_watched_df = movies_watched_df.loc[movies_watched_df.index.isin(users_same_movies), movies_watched]

#Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = movies_watched_df.T.corr().unstack().sort_values(ascending=False).drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ["userId1", "userId2"]

corr_df = corr_df.reset_index()

#Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (>0.65) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_df[(corr_df["userId1"] == random_user) & (corr_df["corr"] >= 0.65)][["userId2", "corr"]].reset_index(drop=True)

top_users.rename(columns={"userId2": "userId"}, inplace=True)

#Adım 4: top_users dataframe'ine rating veri seti ile merge ediniz.
top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

########################################################################
#GÖREV 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
########################################################################

#Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]

#Adım 2: Film id'si ve her bir filme ait tüm kullanıcıların weighted rating'lerinin ortalama değerlerini içeren recommendation_df adında yeni bir dataframe oluşturunuz.
recommendation_df = top_users_rating.groupby("movieId")["weighted_rating"].mean()
recommendation_df = recommendation_df.reset_index()

#Adım 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating'e göre sıralayınız.
recommendation_df = recommendation_df[recommendation_df["weighted_rating"] > 2.5].sort_values(by="weighted_rating", ascending=False)

#Adım 4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.
recommendation_df = recommendation_df.merge(movie[["movieId", "title"]], how="inner")[:5]


#ITEM BASED RECOMMENDATION

########################################################################
#GÖREV 1: Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız
########################################################################
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

#Adım 1: movie, rating datasetlerini okutup birleştiriniz.
movie = pd.read_csv("datasets/movie.csv")
rating = pd.read_csv("datasets/rating.csv")

df = rating.merge(movie, how="left", on="movieId")

df.head()
df.info()

#Adım 2: Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sini alınız.
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

movie_id = df[(df["userId"] == random_user) & (df["rating"] == 5)].sort_values(by="timestamp", ascending=False)["movieId"].iloc[0]

movie_name = df.loc[df["movieId"] == movie_id, "title"].iloc[0]

#Adım 3: User_based_recommendation bölümünde oluşturulan user_movie_df dataframe'ini seçilen film id'sine göre filtreleyiniz.
movie_ratings = um_df[movie_name]

#Adım 4: Filtrelenen dataframe'i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
item_corr = um_df.corrwith(movie_ratings).dropna().sort_values(ascending=False)

recommended_movies = item_corr[1:6]