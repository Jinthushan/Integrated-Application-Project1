from scipy.stats import entropy
from vectorizer import *
from initial_training import *
import numpy as np
from wordcloud import WordCloud

def calc_entropy(prob_dist): #entropy calculating function
    entropy = -np.sum(prob_dist * np.log(prob_dist + (1*(10**-12))),axis=1)
    return entropy

probs_lr = log_model.predict_proba(review_test_vec) #gets probability distributions
probs_nb = nb_model.predict_proba(review_test_vec)

entropy_lr = calc_entropy(probs_lr)
entropy_nb = calc_entropy(probs_nb)

#model confidence distbrution for logistic regression
plt.figure(figsize=(8,5))
sns.histplot(entropy_lr,bins=40,kde=True)
plt.title("Distribution of model Confidence/Uncertainty for Logistic Regression")
plt.xlabel("Entropy")
plt.ylabel("Review Count")
plt.show()

#model confidence distrbution for naive bayes
plt.figure(figsize=(8,5))
sns.histplot(entropy_nb,bins=40,kde=True)
plt.title("Distribution of model Confidence/Uncertainty for Naive Bayes")
plt.xlabel("Entropy")
plt.ylabel("Review Count")
plt.show()

amb_df_lr = pd.DataFrame({'review': review_test.values, 'actual':sen_test.values, 'predicted': sen_pred_lr, 'entropy': entropy_lr}) #ambigious reviews dataframe from logistic regression
amb_df_nb = pd.DataFrame({'review': review_test.values, 'actual':sen_test.values, 'predicted': sen_pred_nb, 'entropy': entropy_nb})


