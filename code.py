from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

sentences=['I have a credit card account','My account card, debit card is lost','My credit card stopped working']

vectorizer=CountVectorizer()
countvec=vectorizer.fit_transform(sentences)

countvec.A

vectorizer.get_feature_names()

vectorizer=CountVectorizer(max_features=4)
countvec=vectorizer.fit_transform(sentences)

print(countvec.A)
print(vectorizer.get_feature_names())

vectorizer=CountVectorizer(max_features=4, stop_words='english')
countvec=vectorizer.fit_transform(sentences)

print(countvec.A)
print(vectorizer.get_feature_names())

vectorizer=CountVectorizer(max_features=6, ngram_range=(1,2))
countvec=vectorizer.fit_transform(sentences)

print(countvec.A)
print(vectorizer.get_feature_names())

vectorizer=TfidfVectorizer(use_idf=False, norm='l1')
tfvec=vectorizer.fit_transform(sentences)

print(tfvec.A)
print(vectorizer.get_feature_names())

vectorizer=TfidfVectorizer(use_idf=False, norm='l2')
tfvec=vectorizer.fit_transform(sentences)

print(tfvec.A)
print(vectorizer.get_feature_names())

print(2/np.sqrt(9))

vectorizer=TfidfVectorizer(use_idf=False, norm=None)
tfvec=vectorizer.fit_transform(sentences)

print(tfvec.A)
print(vectorizer.get_feature_names())

vectorizer_idf=TfidfVectorizer(smooth_idf=False)
tfidfvec=vectorizer_idf.fit_transform(sentences)

print(vectorizer_idf.idf_)
print(vectorizer_idf.get_feature_names())

print(np.log(3/2)+1)

tfidfvec.A
