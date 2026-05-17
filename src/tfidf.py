import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


def preprocess_text(document):
    """Tokeniza, lematiza y elimina stopwords en español."""
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t.lower(), pos='v') for t in tokens]
    return [l for l in lemmas if l not in stopwords.words('spanish')]


def build_tfidf(texts):
    """Ajusta un vectorizador TF-IDF sobre los textos. Retorna (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(analyzer=preprocess_text)
    matrix = vectorizer.fit_transform(pd.DataFrame({'corpus': texts})['corpus'])
    return vectorizer, matrix


def transform_tfidf(vectorizer, texts):
    """Transforma textos con un vectorizador ya ajustado. Retorna matriz sparse."""
    return vectorizer.transform(pd.DataFrame({'corpus': texts})['corpus'])


def display_tfidfs(matrix):
    """Imprime la representación dispersa de la matriz TF-IDF."""
    print(pd.DataFrame.sparse.from_spmatrix(matrix))
