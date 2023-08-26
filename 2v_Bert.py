import pandas as pd
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Загрузка предобученной модели BERT
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Определение весов для компонентов адреса
weights = {
    "street": 1.0,
    "house": 0.8,
    "corpus": 0.5,
    # Добавьте другие части адреса
}

# Загрузка данных
dtypes = {
    "id_building": int,
    "name_prefix": str,
    "house_building": str,
    "corpus_building": str,
    "build_number_building": str,
    "liter_building": str
}

train_df = pd.read_csv('result.csv', dtype=dtypes)
train_df = train_df.replace({np.nan: ""})
train_addresses = train_df['full_address_building'].tolist()

# Преобразование адресов в векторное представление
train_embeddings = model.encode(train_addresses)

# Тренировка модели Nearest Neighbors
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(train_embeddings)

# Загрузка и обработка пользовательских запросов
query_df = pd.read_csv('responseOLD.csv')
query_addresses = query_df['address'].tolist()

# Преобразование адресов запросов в векторное представление
query_embeddings = model.encode(query_addresses)

# Функция для сравнения строк с разными метриками
def similarity_score(str1, str2, metric='cosine'):
    if metric == 'jaccard':
        return fuzz.jaccard(str1, str2)
    elif metric == 'levenshtein':
        return fuzz.ratio(str1, str2)
    elif metric == 'cosine':
        return model.cosine_similarities([model.encode(str1)], [model.encode(str2)])[0]

# Функция для вычисления взвешенной схожести адресов
def weighted_similarity(str1, str2):
    similarity = 0
    for part, weight in weights.items():
        similarity += weight * similarity_score(str1[part], str2[part])
    return similarity

# Выполнение сопоставления адресов
similar_indices = nbrs.kneighbors(query_embeddings, return_distance=False)
matched_addresses = [train_addresses[idx[0]] for idx in similar_indices]
matched_building_ids = [train_df.loc[train_df['full_address_building'] == address, 'id_building'].values[0] for address in matched_addresses]

# Добавление столбцов с результатами в DataFrame запросов
query_df['target_address'] = matched_addresses
query_df['target_building_id'] = matched_building_ids

# Сохранение результатов в файл
query_df.to_csv('populated_response.csv', index=False)
