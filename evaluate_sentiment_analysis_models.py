import os
import pandas as pd
import numpy as np
import random
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
# from nltk.tokenize import sent_tokenize (Tokenization)
from nltk.probability import FreqDist
from nltk.metrics import ConfusionMatrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def _remove_url(data):
    ls = []
    words = ''
    regexp1 = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    regexp2 = re.compile('www?.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    for line in data:
        urls = regexp1.findall(line)

        for u in urls:
            line = line.replace(u, ' ')

        urls = regexp2.findall(line)

        for u in urls:
            line = line.replace(u, ' ')

        ls.append(line)
    return ls


def _remove_regex(data, regex_pattern):
    ls = []
    words = ''

    for line in data:
        matches = re.finditer(regex_pattern, line)

        for m in matches:
            line = re.sub(m.group().strip(), '', line)

        ls.append(line)

    return ls


def _replace_emoticons(data, emoticon_list):
    ls = []

    for line in data:
        for exp in emoticon_list:
            line = line.replace(exp, emoticon_list[exp])

        ls.append(line)

    return ls


def _tokenize_text(data):
    ls = []

    for line in data:
        tokens = wordpunct_tokenize(line)
        ls.append(tokens)

    return ls


def _apply_standardization(tokens, std_list):
    ls = []

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            if word.lower() in std_list:
                word = std_list[word.lower()]

            new_tokens.append(word)

        ls.append(new_tokens)

    return ls


def _remove_stopwords(tokens, stopword_list):
    ls = []

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            if word.lower() not in stopword_list:
                new_tokens.append(word)

        ls.append(new_tokens)

    return ls


def _apply_stemmer(tokens):
    ls = []
    stemmer = nltk.stem.RSLPStemmer()

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            word = str(stemmer.stem(word))
            new_tokens.append(word)

        ls.append(new_tokens)

    return ls


def _get_text_cloud(tokens):
    text = ''

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            text += word + ' '

    return text


def _get_freq_dist_list(tokens):
    ls = []

    for tk_line in tokens:
        for word in tk_line:
            ls.append(word)

    return ls


def _untokenize_text(tokens):
    ls = []

    for tk_line in tokens:
        new_line = ''

        for word in tk_line:
            new_line += word + ' '

        ls.append(new_line)

    return ls


def _get_accuracy(matrix):
    acc = 0
    n = 0
    total = 0

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            if (i == j):
                n += matrix[i, j]

            total += matrix[i, j]

    acc = n / total
    return acc
# predictor
X_col = 'tweet_text'
# classifier
y_col = 'sentiment'

pathTraining = '../DataSet/TrainingDatasets'
pathTest = '../DataSet/TestDatasets'

# print(os.listdir(path))
# Train3Classes.csv

train_ds = pd.read_csv(pathTraining + '/Train3Classes.csv', delimiter=';')

# update classifiers to nominal value
train_ds[y_col] = train_ds[y_col].map({0: 'Negative', 1: 'Positive', 2: 'Neutral'})

X_train = train_ds.loc[:, X_col].values
y_train = train_ds.loc[:, y_col].values

# print(train_ds.sample(5))

# series = train_ds['sentiment'].value_counts()
# ax = series.plot(kind='bar', title='Number for each sentiment')
# ax.set_xlabel('Sentiment')
# ax.set_ylabel('Count')
# plt.show()

# series = train_ds['query_used'].value_counts()
# ax = series.plot(kind='bar', title='Number for each sentiment')
# ax.set_xlabel('Sentiment')
# ax.set_ylabel('Count')
# plt.show()

# check data
# for i in range(0, 5):
#     print(y_train[i], ' => ', X_train[i])

test_ds = pd.read_csv(pathTest + '/Test3classes.csv', delimiter=';')

# update classifiers to nominal value
test_ds[y_col] = test_ds[y_col].map({0: 'Negative', 1: 'Positive', 2: 'Neutral'})

X_test = test_ds.loc[:, X_col].values
y_test = test_ds.loc[:, y_col].values

X_train = _remove_url(X_train)
X_test = _remove_url(X_test)

# hashtags
regex_pattern = '#[\w]*'
X_train = _remove_regex(X_train, regex_pattern)
X_test = _remove_regex(X_test, regex_pattern)

# notations
regex_pattern = '@[\w]*'
X_train = _remove_regex(X_train, regex_pattern)
X_test = _remove_regex(X_test, regex_pattern)

# for i in range(0, 5):
#     print(X_train[i])

emoticon_list = {':))': 'positive_emoticon', ':)': 'positive_emoticon', ':D': 'positive_emoticon', ':(': 'negative_emoticon', ':((': 'negative_emoticon', '8)': 'neutral_emoticon'}

X_train = _replace_emoticons(X_train, emoticon_list)
X_test = _replace_emoticons(X_test, emoticon_list)

X_train_tokens = _tokenize_text(X_train)
X_test_tokens = _tokenize_text(X_test)

# create your own list
std_list = {'eh': 'é', 'vc': 'você', 'vcs': 'vocês','tb': 'também', 'tbm': 'também', 'obg': 'obrigado', 'gnt': 'gente', 'q': 'que', 'n': 'não', 'cmg': 'comigo', 'p': 'para', 'ta': 'está', 'to': 'estou', 'vdd': 'verdade'}

X_train_tokens = _apply_standardization(X_train_tokens, std_list)
X_test_tokens = _apply_standardization(X_test_tokens, std_list)

# print(X_test_tokens[0:5])

stopword_list = []
nltk_stopwords = nltk.corpus.stopwords.words('portuguese')
stopword_list.append('é')
stopword_list.append('vou')
stopword_list.append('que')
stopword_list.append('tão')
stopword_list.append('...')
stopword_list.append('«')
stopword_list.append('➔')
stopword_list.append('|')
stopword_list.append('»')
stopword_list.append('uai') # 'expression from the mineiros (MG/Brazil)'
noisy_words = ['.', '?', '!', ':', ',', ';', '(', ')', '-']
stopword_list.extend(nltk_stopwords)
stopword_list.extend(noisy_words)
stopword_list = list(set(stopword_list))

X_train_tokens = _remove_stopwords(X_train_tokens, stopword_list)
X_test_tokens = _remove_stopwords(X_test_tokens, stopword_list)

# for i in range(0, 5):
#     print(X_train_tokens[i])

X_train_tokens = _apply_stemmer(X_train_tokens)
X_test_tokens = _apply_stemmer(X_test_tokens)

# for i in range(0, 5):
#     print(X_train_tokens[i])

# print train WordCloud
# sample_train = random.sample(X_train_tokens, 10000)
# text_cloud = _get_text_cloud(sample_train)
#
# word_cloud = WordCloud(max_font_size = 100, width = 1520, height = 535)
# word_cloud.generate(text_cloud)
# plt.figure(figsize = (16, 9))
# plt.imshow(word_cloud)
# plt.axis('off')
# plt.show()

# print test WordCloud
# sample_test = random.sample(X_test_tokens, len(X_test_tokens))
# text_cloud = _get_text_cloud(sample_test)
#
# word_cloud = WordCloud(max_font_size = 100, width = 1520, height = 535)
# word_cloud.generate(text_cloud)
# plt.figure(figsize = (16, 9))
# plt.imshow(word_cloud)
# plt.axis('off')
# plt.show()

# Frequency Distribution on training dataset
# fd_list = _get_freq_dist_list(X_train_tokens)
# fdist = FreqDist(fd_list)
# print(fdist)
#
# most_common = fdist.most_common(25)
# print(most_common)
#
# most_uncommon = fdist.hapaxes()
# print(most_uncommon[0:30])
#
# # find the word occuring max number of times
# print(fdist.max())

# print most common words
# series = pd.Series(data=[v for k, v in most_common], index=[k for k, v in most_common], name='')
# ax = series.plot(kind='bar', title='Frequency Distribution')
# ax.set_xlabel('Word')
# ax.set_ylabel('Count')
# plt.show()

X_train = _untokenize_text(X_train_tokens)
X_test = _untokenize_text(X_test_tokens)

vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_vect)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

new_corpus = [
        '@acme A alegria está na luta, na tentativa, no sofrimento envolvido e não na vitória propriamente dita!',
        'A alegria evita mil males e prolonga a vida.',
        'não se deve maltratar os idosos, eles possuem muita sabedoria!',
        '#filmedevampiro tome muito cuidado com o dracula... :( www.filmedevampiro.com.br'
        ]

X_new = new_corpus

# Remove urls from text (http(s), www)
X_new = _remove_url(X_new)

# Remove hashtags
regex_pattern = '#[\w]*'
X_new = _remove_regex(X_new, regex_pattern)

# Remove notations
regex_pattern = '@[\w]*'
X_new = _remove_regex(X_new, regex_pattern)

# Replace emoticons ":)) :) :D :(" to positive_emoticon or negative_emoticon or neutral_emoticon
X_new = _replace_emoticons(X_new, emoticon_list)

# Tokenize text
X_new_tokens = _tokenize_text(X_new)

# Object Standardization
X_new_tokens = _apply_standardization(X_new_tokens, std_list)

# remove stopwords
X_new_tokens = _remove_stopwords(X_new_tokens, stopword_list)

# Stemming (dimensionality reduction)
X_new_tokens = _apply_stemmer(X_new_tokens)

# Dataset preparation
# Untokenize text (transform tokenized text into string list)
X_new = _untokenize_text(X_new_tokens)

# Text to Features
# Feature extraction from text
# Method: bag of words
X_new_vect = vectorizer.transform(X_new)

# TF-IDF: Term Frequency - Inverse Document Frequency
# use the transform(...) method to transform count-matrix to a tf-idf representation.
X_new_tfidf = tfidf_transformer.transform(X_new_vect)

standalone_predictions = model.predict(X_new_tfidf)

# for doc, prediction in zip(new_corpus, standalone_predictions):
#     print('%r => %s' % (doc, prediction))

# Text to Features
# Feature extraction from text
# Method: bag of words
X_test_vect = vectorizer.transform(X_test)
# print(X_test_vect.shape)

X_test_tfidf = tfidf_transformer.transform(X_test_vect)
# print(X_test_tfidf.shape)

# predictions = model.predict(X_test_tfidf)
#
# matrix = metrics.confusion_matrix(y_test, predictions)
#
# print(model.classes_)
#
# acc1 = np.mean(predictions == y_test)
# acc2 = _get_accuracy(matrix)
# print(acc1, acc2)

# print(metrics.classification_report(y_test, predictions, target_names=model.classes_))

model_LR = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')),
])

model_LR.fit(X_train, y_train)
predictions_LR = model_LR.predict(X_test)
joblib.dump(model_LR, '../LogisticRegression_model.joblib')

matrix = metrics.confusion_matrix(y_test, predictions_LR)
acc = _get_accuracy(matrix)
print('LogisticRegression_model: ' + str(acc))

model_MNB = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

model_MNB.fit(X_train, y_train)
predictions_MNB = model_MNB.predict(X_test)

joblib.dump(model_MNB, '../MultinomialNB_model.joblib')

matrix = metrics.confusion_matrix(y_test, predictions_MNB)
acc = _get_accuracy(matrix)
print('MultinomialNB_model: ' + str(acc))


model_SGD = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 1e-3, random_state = 42, max_iter = 5, tol = None)),
])

model_SGD.fit(X_train, y_train)
predictions_SGD = model_SGD.predict(X_test)

joblib.dump(model_MNB, '../SGDClassifier_model.joblib')

matrix = metrics.confusion_matrix(y_test, predictions_SGD)
acc = _get_accuracy(matrix)
print('SGDClassifier_model: ' + str(acc))