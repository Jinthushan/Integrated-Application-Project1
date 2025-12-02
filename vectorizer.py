import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(base_dir + r"\imdb_dataset.csv") #dataset
df['review'] = df['review'].str.lower().str.replace('[,?.!"]', '', regex=True) #cleaning text

review = df['review']
sen = df['sentiment']

#splitting data for training and testing
review_train, review_test, sen_train, sen_test = train_test_split(review,sen, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

review_train_vec = vectorizer.fit_transform(review_train)
review_test_vec = vectorizer.transform(review_test)