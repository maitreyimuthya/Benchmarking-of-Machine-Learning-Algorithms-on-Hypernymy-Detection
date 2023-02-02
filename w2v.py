from gensim.models import Word2Vec
import pandas as pd


def run_word2vec_on_column(col):
    return Word2Vec([[x] for x in col], vector_size=50, min_count=1)


def word2vec_embeddings(data_df):
    x_embeddings = run_word2vec_on_column(data_df['x'])
    y_embeddings = run_word2vec_on_column(data_df['y'])

    data_df = data_df.reset_index()

    embedding_matrix = []
    for index, row in data_df.iterrows():
        embedding_matrix.append([x_embeddings.wv.vectors[x_embeddings.wv.key_to_index[row['x']]],
                                 y_embeddings.wv.vectors[y_embeddings.wv.key_to_index[row['y']]],
                                 int(row['label'])])

    return pd.DataFrame(data=embedding_matrix, columns=['x', 'y', 'label'])
