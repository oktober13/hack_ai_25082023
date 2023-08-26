import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

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
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_addresses)

# Тренировка модели Nearest Neighbors
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(X_train)

# Загрузка и обработка пользовательских запросов
query_df = pd.read_csv('response.csv')
query_addresses = query_df['address'].tolist()
X_query = vectorizer.transform(query_addresses)

# Выполняем сопоставление адресов
similar_indices = nbrs.kneighbors(X_query, return_distance=False)
matched_addresses = [train_addresses[idx[0]] for idx in similar_indices]
matched_building_ids = [train_df.loc[train_df['full_address_building'] == address, 'id_building'].values[0] for address in matched_addresses]

# заполняем набор данных ответов
query_df['target_address'] = matched_addresses
query_df['target_building_id'] = matched_building_ids

# Сохраняем результирующий датасет
query_df.to_csv('populated_response_v1.csv', index=False)