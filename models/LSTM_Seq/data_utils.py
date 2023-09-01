import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    column_names = ["user_id", "movie_id", "rating", "timestamp"]
    return pd.read_csv(filepath, sep="::", names=column_names, engine='python')

def generate_train_sequences(ratings_df, SEQUENCE_LENGTH):
    user_sequences = ratings_df.groupby('user_id')['movie_id'].apply(list).tolist()
    
    user_ids, sequence_data = [], []
    for user_id, seq in enumerate(user_sequences, 1):
        if len(seq) >= SEQUENCE_LENGTH:
            user_ids.append(user_id)
            sequence_data.append(seq[-SEQUENCE_LENGTH:])
    return np.array(user_ids), np.array(sequence_data)

def prepare_data(filepath, SEQUENCE_LENGTH, test_size=0.2, random_state=42):
    ratings_df = load_data(filepath)
    train_ratings, val_ratings = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
    train_users, train_sequences = generate_train_sequences(train_ratings, SEQUENCE_LENGTH)
    val_users, val_sequences = generate_train_sequences(val_ratings, SEQUENCE_LENGTH)
    return train_sequences, val_sequences, ratings_df['movie_id'].max(), ratings_df['user_id'].max()
