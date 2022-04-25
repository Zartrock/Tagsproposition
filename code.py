
import pickle
f = open("test.txt", "rb")
print(f.read())

f2 = open("lda_model.pkl", "rb")
lda_model = pickle.load(f2)
print('good')

f1 = open("dictionary.pkl", "rb")
id2word = pickle.load(f1)


from sklearn.model_selection import train_test_split
import pandas as pd
import lxml
import html5lib
import re
import pickle
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.utils import simple_preprocess
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel

nltk.download('wordnet', r'C:\Users\jonas\PycharmProjects\pythonProject1\code\nltk_data')
nltk.data.path.append(r'C:\Users\jonas\PycharmProjects\pythonProject1\code\nltk_data')

##'''with open(r'C:/Users/jonas/Desktop/OC/P5/code/lda_model.pkl', 'rb') as file:
##   lda_model = pickle.load(file)

##with open(r'C:/Users/jonas/Desktop/OC/P5/code/dictionary.pkl', 'rb') as filevoc:
##    id2word = pickle.load(filevoc)''')

def clean_html(text):
    """
    Remove HTML from a text.

    Args:
        text(String): Row text with html
    Returns:
        cleaned String
    """
    soup = BeautifulSoup(text, "html5lib")

    for sent in soup(['style', 'script']):
        sent.decompose()

    return ' '.join(soup.stripped_strings)


def text_cleaning(text):
    """
    Remove figures, punctuation, words shorter than two letters (excepted C or R) in a lowered text.

    Args:
        text(String): Row text to clean
    Returns:
        res(string): Cleaned text
    """
    pattern = re.compile(r'[^\w]|[\d_]')

    try:
        res = re.sub(pattern, " ", text).lower()
    except TypeError:
        return text

    res = res.split(" ")
    res = list(filter(lambda x: len(x) > 3, res))  # Keep singles c and r because it might be used as name of languages
    res = " ".join(res)
    return res


def tokenize(text):
    """
    Tokenize words of a text.

    Args:
        text(String): Row text
    Returns
        res(list): Tokenized string.
    """
    stop_words = set(stopwords.words('english'))

    try:
        res = word_tokenize(text, language='english')
    except TypeError:
        return text

    res = [token for token in res if token not in stop_words]
    return res


def filtering_nouns(tokens):
    """
    Filter singular nouns

    Args:
        tokens(list): A list o tokens

    Returns:

        res(list): Filtered token list
    """
    res = nltk.pos_tag(tokens)

    res = [token[0] for token in res if token[1] == 'NN']

    return res


def lemmatize(tokens):
    """
    Transform tokens into lems

    Args:
        tokens(list): List of tokens
    Returns:
        lemmatized(list): List of lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized = []

    for token in tokens:
        lemmatized.append(lemmatizer.lemmatize(token))

    return lemmatized


class LdaModel:

    def __init__(self):
        filename_model = r"C:\Users\sesa638933\Desktop\OC\P5\lda_model.pkl"
        filename_dictionary = r"C:\Users\sesa638933\Desktop\OC\P5\dictionary.pkl"
        self.model = pickle.load(open(filename_model, 'rb'))
        self.dictionary = pickle.load(open(filename_dictionary, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags
        zof a preprocessed text

        Args:
            text(list): preprocessed text
        Returns:
            res(list): list of tags
        """
        corpus_new = self.dictionary.doc2bow(text)
        topics = self.model.get_document_topics(corpus_new)

        # find most relevant topic according to probability
        relevant_topic = topics[0][0]
        relevant_topic_prob = topics[0][1]

        for i in range(len(topics)):
            if topics[i][1] > relevant_topic_prob:
                relevant_topic = topics[i][0]
                relevant_topic_prob = topics[i][1]

        # retrieve associated to topic tags present in submited text
        res = self.model.get_topic_terms(topicid=relevant_topic, topn=20)

        res = [self.dictionary[tag[0]] for tag in res if self.dictionary[tag[0]] in text]

        return res


def predict_unsupervised_tags(text):
    """
    Predict tags of a preprocessed text

    Args:
        text(list): preprocessed text

    Returns:
        relevant_tags(list): list of tags
    """

    corpus_new = id2word.doc2bow(text)
    topics = lda_model.get_document_topics(corpus_new)

    # find most relevant topic according to probability
    relevant_topic = topics[0][0]
    relevant_topic_prob = topics[0][1]

    for i in range(len(topics)):
        if topics[i][1] > relevant_topic_prob:
            relevant_topic = topics[i][0]
            relevant_topic_prob = topics[i][1]

    # retrieve associated to topic tags present in submited text
    potential_tags = lda_model.get_topic_terms(topicid=relevant_topic, topn=20)

    relevant_tags = [id2word[tag[0]] for tag in potential_tags if id2word[tag[0]] in text]

    return relevant_tags


from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import streamlit as st

text = st.text_input('Your question:', 'write your question here')

text_wo_html = clean_html(text)
cleaned_text = text_cleaning(text_wo_html)
tokenized_text = tokenize(cleaned_text)
filtered_noun_text = filtering_nouns(tokenized_text)
lemmatized_text = lemmatize(filtered_noun_text)
unsupervised_tags = predict_unsupervised_tags(lemmatized_text)

st.write('The tags proposed are:', unsupervised_tags)
