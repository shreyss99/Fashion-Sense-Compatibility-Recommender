import pickle
from annoy import AnnoyIndex


class FashionRecommender:
    def __init__(self, feature_embeddings_file, image_names_file):
        self.feature_embeddings = pickle.load(open(feature_embeddings_file, 'rb'))
        self.image_names = pickle.load(open(image_names_file, 'rb'))
        self.embedding_size = len(self.feature_embeddings[0])
        self.num_trees = 50
        self.annoy_index = AnnoyIndex(self.embedding_size, 'angular')  # Use cosine similarity

        # Build Annoy index
        for i, embedding in enumerate(self.feature_embeddings):
            self.annoy_index.add_item(i, embedding)
        self.annoy_index.build(self.num_trees)

    def recommend_similar_images(self, image_features, num_results=5):
        similar_image_indices = self.annoy_index.get_nns_by_vector(image_features, num_results)
        similar_image_paths = [self.image_names[idx] for idx in similar_image_indices]
        return similar_image_paths
