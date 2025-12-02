from vectorizer import review_train_vec, review_test_vec, sen_train, sen_test
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#logistic regression model
log_model = LogisticRegression(max_iter=1000, solver='liblinear') 
log_model.fit(review_train_vec, sen_train)
sent_pred_lr = log_model.predict(review_test_vec)

#naive bayes model
nb_model = MultinomialNB()
nb_model.fit(review_train_vec, sen_train)
sen_pred_nb = nb_model.predict(review_test_vec)

#Classification Reports of both models
print("Logistic Regression Accuracy:", accuracy_score(sen_test, sent_pred_lr))
print(classification_report(sen_test, sent_pred_lr))

print("Naive Bayes Accuracy:", accuracy_score(sen_test, sen_pred_nb))
print(classification_report(sen_test, sen_pred_nb))
