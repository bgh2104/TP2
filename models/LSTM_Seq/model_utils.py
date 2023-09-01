from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Dense
from keras.regularizers import l2
from sklearn.metrics import ndcg_score
import numpy as np

def build_model(SEQUENCE_LENGTH, num_users, num_movies, EMBEDDING_DIM):
    user_input = Input(shape=(SEQUENCE_LENGTH,), dtype='int32', name='user_sequence_input')
    user_embedding = Embedding(num_users + 1, EMBEDDING_DIM, input_length=SEQUENCE_LENGTH, name='user_embedding', embeddings_regularizer=l2(0.001))(user_input)

    lstm_out = LSTM(100)(user_embedding)
    lstm_out = Dropout(0.5)(lstm_out)
    output = Dense(num_movies + 1, activation='softmax')(lstm_out)

    model = Model(inputs=user_input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_sequences, num_movies, epochs=5, batch_size=64):
    train_labels = np.zeros((train_sequences.shape[0], num_movies + 1))
    for idx, seq in enumerate(train_sequences):
        train_labels[idx, seq[-1]] = 1
    model.fit(train_sequences, train_labels, epochs=epochs, batch_size=batch_size)

def evaluate_model(model, sequences, N=10):
    precisions, recalls, ndcgs = [], [], []
    for seq in sequences:
        true_movie = seq[-1]
        predictions = model.predict(np.array([seq]))
        recommended_movies = top_n_recommendation(model, seq, N)

        precisions.append(1 if true_movie in recommended_movies else 0)
        recalls.append(1 if true_movie in recommended_movies[:len(seq)] else 0)

        true_relevance = np.zeros(N)
        true_relevance[0] = 1 if true_movie == recommended_movies[0] else 0
        ndcgs.append(ndcg_score([true_relevance], [predictions[0][recommended_movies]]))

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

def top_n_recommendation(model, sequence, num_movies=5):
    predictions = model.predict(np.array([sequence]))
    recommended_indices = np.argsort(-predictions[0])[:num_movies]
    return recommended_indices
