import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json


class AddressMatcher:
    def __init__(self, weights):
        self.weights = weights
        self.vectorizer = TfidfVectorizer()
        self.nbrs = None
        self.train_df = None
        self.train_addresses = None
        self.X_train = None
        self.query_addresses = None
        self.X_query = None
        self.matched_addresses = None
        self.matched_building_ids = None

    def load_train_data(self, train_data_path):
        dtypes = {
            "id_building": int,
            "post_prefix_building": object,
            "full_address_building": str,
            "name_prefix": str,
            "house_building": str,
            "corpus_building": str,
            "build_number_building": str,
            "liter_building": str
        }
        self.train_df = pd.read_csv(train_data_path, dtype=dtypes, low_memory=False)
        self.train_df = self.train_df.replace({np.nan: ""})
        self.train_addresses = self.train_df['full_address_building'].tolist()
        self.X_train = self.vectorizer.fit_transform(self.train_addresses)
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(self.X_train)

    def load_query_data(self, query_data_path):
        query_df = pd.read_csv(query_data_path)
        self.query_addresses = query_df['address'].tolist()
        self.X_query = self.vectorizer.transform(self.query_addresses)

    def match_addresses(self):
        similar_indices = self.nbrs.kneighbors(self.X_query, return_distance=False)
        matched_addresses = [self.train_addresses[idx[0]] for idx in similar_indices]
        matched_building_ids = [
            self.train_df.loc[self.train_df['full_address_building'] == address, 'id_building'].values[0] for address in
            matched_addresses]
        return matched_addresses, matched_building_ids

    def calculate_weighted_similarity(self):
        weighted_similarities = []
        for query, matched in zip(self.query_addresses, self.matched_addresses):
            query_parts = query.split()
            matched_parts = matched.split()
            weighted_similarity = 0
            for part, weight in self.weights.items():
                if part in query_parts and part in matched_parts:
                    similarity = np.dot(self.X_query[0, self.vectorizer.vocabulary_[part]],
                                        self.X_train[0, self.vectorizer.vocabulary_[part]])
                    weighted_similarity += similarity * weight
            weighted_similarities.append(weighted_similarity)
        return weighted_similarities

    def run(self, train_data_path, query_data_path):
        self.load_train_data(train_data_path)
        self.load_query_data(query_data_path)
        self.matched_addresses, self.matched_building_ids = self.match_addresses()
        weighted_similarities = self.calculate_weighted_similarity()

        query_df = pd.read_csv(query_data_path)
        query_df['target_address'] = self.matched_addresses
        query_df['target_building_id'] = self.matched_building_ids
        query_df['weighted_similarity'] = weighted_similarities

        query_df = query_df.sort_values(by='weighted_similarity', ascending=False)
        query_df.to_csv('populated_response_v3.csv', index=False)

    def find_matching_address(self, query_address):
        query_address = [query_address]
        X_query = self.vectorizer.transform(query_address)
        similar_indices = self.nbrs.kneighbors(X_query, return_distance=False)
        matched_address = self.train_addresses[similar_indices[0][0]]
        matched_building_id = int(
            self.train_df.loc[self.train_df['full_address_building'] == matched_address, 'id_building'].values[0])

        response = {
            "success": True,
            "query": [{"address": query_address[0]}],
            "result": [
                {
                    "target_building_id": matched_building_id,
                    "target_address": matched_address
                }
            ]
        }

        return json.dumps(response, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    weights = {
        "post_prefix_building": 0.9,
        "full_address_building": 0.8,
        "name_prefix": 0.6,
        "house_building": 0.7,
        "corpus_building": 0.4,
        "build_number_building": 0.5,
        "liter_building": 0.1
    }

    address_matcher = AddressMatcher(weights)
    address_matcher.load_train_data('result.csv')

    # Для Александра пример использования для поиска адреса и создания JSON-ответа
    query_address = "аптерский 18 спб"
    json_response = address_matcher.find_matching_address(query_address)
    print(json_response)


