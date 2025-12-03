import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def negationHandling(words): #negation handling
    negation_words = {
        "not", "no", "never", "nothing",
        "don't", "doesn't", "didn't", "can't", "couldn't",
        "won't", "shouldn't", "isn't", "aren't", "wasn't", "weren't"
    }

    flip_words = {"but", "however", "though", "yet"} #for terms that cancel out negation

    negation_active = False
    neg_count = 0
    max_neg = 5

    result = []
    for w in words:

        if w in negation_words:
            negation_active = True
            neg_count = 0
            result.append(w)
            continue

        if w in flip_words:
            negation_active = False
            result.append(w)
            continue

        if negation_active:
            if len(w) > 2:
                w = w + "_NEG" 
            neg_count += 1
            if neg_count >= max_neg:
                negation_active = False
            result.append(w)
        else:
            result.append(w)

    return " ".join(result)

def applyNegHandling(text):
    words = text.split()
    return negationHandling(words)

base_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(base_dir + r"\imdb_dataset.csv") #dataset


df['review'] = df['review'].str.lower().str.replace('[,?.!"]', '', regex=True) #cleaning text
df['review'] = df['review'].apply(applyNegHandling)

review = df['review']
sen = df['sentiment']

#splitting data for training and testing
review_train, review_test, sen_train, sen_test = train_test_split(review,sen, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2) ,max_features=10000) #tfidf vectorizer with n-grams to improve performance

review_train_vec = vectorizer.fit_transform(review_train)
review_test_vec = vectorizer.transform(review_test)