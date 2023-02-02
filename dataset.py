import numpy as np
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

from w2v import word2vec_embeddings
import nltk
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')


def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}', "")
    cleaner = re.compile('<.*?>')
    cleantext = re.sub(cleaner, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if w not in stopwords.words('english')]
    # stem_words = [nltk.stem.porter.PorterStemmer().stem(w) for w in filtered_words]
    # lemma_words = [nltk.stem.wordnet.WordNetLemmatizer().lemmatize(w) for w in stem_words]
    return "_".join(filtered_words)


def read_data(filepath, sep='\t'):
    data_df = pd.read_csv(filepath, sep=sep, names=['x', 'y', 'label'])
    data_df['x'] = [preprocess(x) for x in data_df['x']]
    data_df['y'] = [preprocess(y) for y in data_df['y']]
    return data_df


def get_train_test_valid_data():
    training_embeddings = word2vec_embeddings(read_data(filepath='train.tsv'))
    test_embeddings = word2vec_embeddings(read_data(filepath='test.tsv'))
    validation_embeddings = word2vec_embeddings(read_data(filepath='val.tsv'))

    x_train, y_train, x_test, y_test, x_valid, y_valid = np.array(
        [[row['x'], row['y']] for idx, row in training_embeddings.iterrows()]), np.array(
        [int(x) for x in training_embeddings['label']]), np.array(
        [[row['x'], row['y']] for idx, row in test_embeddings.iterrows()]), np.array(
        [int(x) for x in test_embeddings['label']]), np.array(
        [[row['x'], row['y']] for idx, row in validation_embeddings.iterrows()]), np.array(
        [int(x) for x in validation_embeddings['label']])

    return reshape(x_train), y_train, reshape(x_test), y_test, reshape(x_valid), y_valid


def reshape(x):
    return x.reshape(len(x), len(x[0]) * len(x[0][0]))
