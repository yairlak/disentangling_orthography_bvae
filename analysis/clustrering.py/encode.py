from itertools import combinations

'''
Generate embeddings of pairwise bigrams

Types of encodings: 
    encode_4_bytes: one-hot encoding based on a equality matrix
        From the bigram b=("AB","CD"), the resulting embedding is
            e = ["A"=="C", "A"=="D", "B"=="C", "B"=="D"]
    
    encode_letters_by_4_bytes: one hot enconing based on the presence of each letter in the pairwise bigram
        From the bigram b=("AB","CD"), with A,B,C,D the only possible chars, the resulting embedding is
            e = ["A"=="A", "A"=="B", "A"=="C", "A"=="D",
                 "B"=="A", "B"=="B", "B"=="C", "B"=="D",
                 "C"=="A", "C"=="B", "C"=="C", "C"=="D",
                 "D"=="A", "D"=="B", "D"=="C", "D"=="D"]        
    
    encode_4_bytes_dist: similar to encode_4_bytes but using the distance of the model for each unigram
        From the bigram b=("AB","CD"), the resulting embedding is
                e = [dist("A","C"), dist("A","D"), dist("B","C"), dist("B","D")]
'''


def encode_4_bytes(letters):
    pairs = list(combinations(letters, 2))
    n_bytes = 4
    embs = []
    for pair in pairs:
        emb = [0]*(n_bytes)

        emb[0] = int(pair[0][0] == pair[1][0])
        emb[1] = int(pair[0][0] == pair[1][1])
        emb[2] = int(pair[0][1] == pair[1][0])
        emb[3] = int(pair[0][1] == pair[1][1])

        embs.append(emb)
    return embs


def encode_letters_by_4_bytes(letters):
    pairs = list(combinations(letters, 2))
    set_letters = list(set("".join(letters)))
    set_letters.sort()

    n_bytes = 4*len(set_letters)
    embs = []
    for pair in pairs:
        emb = [0]*(n_bytes)
        for i, l in enumerate(set_letters):
            pos = i*4
            emb[pos+0] = int(pair[0][0] == l)
            emb[pos+1] = int(pair[0][1] == l)
            emb[pos+2] = int(pair[1][0] == l)
            emb[pos+3] = int(pair[1][1] == l)

        embs.append(emb)
    return embs


def encode_4_bytes_dist(letters_bigram, letters_unigram, dist_unigrams ):

    pairs = list(combinations(letters_bigram, 2))
    n_bytes = 4
    embs = []
    for pair in pairs:
        emb = [0]*(n_bytes)

        index_A = letters_unigram.index(pair[0][0])
        index_B = letters_unigram.index(pair[0][1])
        index_C = letters_unigram.index(pair[1][0])
        index_D = letters_unigram.index(pair[1][1])

        emb[0] = dist_unigrams[index_A][index_C]
        emb[1] = dist_unigrams[index_A][index_D]
        emb[2] = dist_unigrams[index_B][index_C]
        emb[3] = dist_unigrams[index_B][index_D]

        embs.append(emb)
    return embs

