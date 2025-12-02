import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud



df = pd.read_csv(r'IMDB_Datset.csv') #dataset
df['review'] = df['review'].str.lower().str.replace('[,?.!"]', '', regex=True) #cleaning text

review = df['review']
sent = df['sentiment']

#splitting data for training and testing
review_train, review_test, sent_train, sent_test = train_test_split(review,sent, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='enlgish', max_features=10000)

review_train_vec = vectorizer.fit_transform(review_train)
review_test_vec = vectorizer.fit_transform(review_test)