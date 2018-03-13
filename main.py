# -*- coding: UTF-8 -*-.
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

STOP_WORDS="english"
# STOP_WORDS=None
MAX_FEATURES = 1000
MIN_DF = 0.1
MAX_DF = 0.5

# read csv input file
def read_data(input_file):
    with open(input_file, 'r') as file:
        # fields are: label, title and text
        reader = csv.DictReader(file, fieldnames=["label", "title", "text"])
        # initialize texts and labels arrays
        texts = []
        labels = []
        # iterate over file rows
        for row in reader:
            # append label and texts
            labels.append(int(row["label"]))
            texts.append(row["text"])
        return labels, texts

# main program
def main():
    # open test and train data
    test_labels, test_texts = read_data('db/ag_news_csv/test.csv')
    train_labels, train_texts = read_data('db/ag_news_csv/train.csv')

    # initialize tfidf vectorizer
    tfidf = TfidfVectorizer(strip_accents="ascii",stop_words=STOP_WORDS,max_features=MAX_FEATURES)
    # fit tfidf with train texts
    tfidf.fit(train_texts)
    # transform train and test texts to numerical features
    train_features = tfidf.transform(train_texts)
    test_features = tfidf.transform(test_texts)

    clf = LinearSVC()
    clf.fit(train_features,train_labels)
    pred = clf.predict(test_features)
    print "Accuracy:", accuracy_score(test_labels, pred)

if __name__ == "__main__":
    main()
