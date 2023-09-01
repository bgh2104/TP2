import data_utils
import model_utils

EMBEDDING_DIM = 50
SEQUENCE_LENGTH = 5

train_sequences, val_sequences, num_movies, num_users = data_utils.prepare_data('../../data/ratings.dat', SEQUENCE_LENGTH)

model = model_utils.build_model(SEQUENCE_LENGTH, num_users, num_movies, EMBEDDING_DIM)
model_utils.train_model(model, train_sequences, num_movies)

precision, recall, ndcg = model_utils.evaluate_model(model, val_sequences)
print(f"Precision@10{SEQUENCE_LENGTH}: {precision:.4f}")
print(f"Recall@10{SEQUENCE_LENGTH}: {recall:.4f}")
print(f"NDCG@10{SEQUENCE_LENGTH}: {ndcg:.4f}")
