import numpy as np
import pandas as pd
from als import ExplicitMF


def display_top_k_movies(similarity, movies, movie_idx):
    movie_indices = np.argsort(similarity[movie_idx])
    i = 1
    while i < 4:
        print(movies[movie_indices[i]])
        i += 1

def compare_recs(als_similarity, movies,\
                 movie_idx):
    print(movies[movie_idx])
    display_top_k_movies(als_similarity, idx_to_movie,\
                         movie_idx)

def cosine_similarity(model):
    sim = model.item_vecs.dot(model.item_vecs.T)
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return sim / norms / norms.T


names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=names)

ratings = np.empty((943, 1682))

for index, row in df.iterrows():
    ratings[row[0]-1, row[1]-1] = row[2]


# Load in movie data
idx_to_movie = {}
with open('u.item', encoding="UTF-16") as f:
    for line in f.readlines():
        info = line.split('|')
        idx_to_movie[int(info[0])-1] = info[1]

best_als_model = ExplicitMF(ratings)
best_als_model.train(100)

als_sim = cosine_similarity(best_als_model)



idx = 0 # Toy Story
compare_recs(als_sim, idx_to_movie, idx)

# idx =idx 26 # Bad Badoys
# compare_recs(als_sim, idx_to_movie, idx)
#

# idx = 28 #idx_to_moviedx Batman Forever
# compare_recs(als_sim, idx_to_movie, idx)
#

# idx = 500 # Dumbo
# compare_recs(als_sim, idx_to_movie, idx)