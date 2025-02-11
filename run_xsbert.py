import torch
from sentence_transformers.models import Pooling
from xsbert import models, utils
import pandas as pd
from collections import defaultdict
import pickle
from tqdm import tqdm
import time
from datetime import timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformer = models.ReferenceTransformer('tceron/sentence-transformers-party-similarity-by-domain')
transformer = models.ReferenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
pooling = Pooling(transformer.get_word_embedding_dimension())
model = models.XSRoberta(modules=[transformer, pooling]) 
model.to(torch.device('cuda:1'))

model.reset_attribution()
model.init_attribution_to_layer(idx=10, N_steps=100)

def generate_explanation(texta, textb):
    A, tokens_a, tokens_b = model.explain_similarity(
        texta, 
        textb, 
        move_to_cpu=True,
        sim_measure='cos',
    )
    return A, tokens_a, tokens_b

def retrieve_tokens_high_similarity(A, tokens_a, tokens_b, k):
    """ Retrieve indexes from texta and textb with similarity higher than k according to the attributions in matrix A """
    list_tokens = []
    for i, token_a in enumerate(tokens_a):
        for j, token_b in enumerate(tokens_b):
            sim = A[i, j]
            if sim > k:
                max_sim_idx = (i, j)
                list_tokens.append((token_a, token_b))

    return list_tokens

df = pd.read_csv('../capture_similarity_between_political_parties/data/sentence_pairs.csv', index_col=0)
print(len(df))

partya= "gruene"
partyb= "fdp"
df = df[(df["party1"]==partya)&(df["party2"]==partyb)]
print(len(df))

start_time = time.time()
party_tokens = defaultdict(list)

# Add tqdm to show progress bar with estimated time
for i, row in tqdm(enumerate(df.iterrows(), 1), total=len(df)):
    texta = row[1].sentence1
    textb = row[1].sentence2
    
    A, tokens_a, tokens_b = generate_explanation(texta, textb)
    similar_tokens = retrieve_tokens_high_similarity(A, tokens_a, tokens_b, k=0.01)
    party_tokens[(partya, partyb)].extend(similar_tokens)

        # Optional: Print estimated time remaining periodically
    if i % 10 == 0:
        elapsed = time.time() - start_time
        estimated_total = (elapsed / i) * len(df)
        remaining = estimated_total - elapsed
        
        print(f"\nElapsed: {timedelta(seconds=int(elapsed))}")
        print(f"Estimated remaining: {timedelta(seconds=int(remaining))}")

# Optional: Print total processing time
total_time = time.time() - start_time
print(f"\nTotal processing time: {total_time:.2f} seconds")

# save party_tokens in pickle
with open(f'party_tokens_{partya}_{partyb}.pkl', 'wb') as f:
    pickle.dump(party_tokens, f)