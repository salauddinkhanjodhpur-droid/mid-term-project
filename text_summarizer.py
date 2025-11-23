import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re


# --------- Article ko read karke sentences list banana ---------
def read_article(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

    # Simple sentence split
    raw_sentences = [s.strip() for s in text.split(". ") if s.strip()]

    sentences = []
    for sentence in raw_sentences:
        # Non-letters hatao
        cleaned = re.sub(r"[^a-zA-Z]", " ", sentence)
        words = cleaned.split()
        if words:
            sentences.append(words)

    return sentences


# --------- Do sentences ka similarity nikalna ---------
def sentence_similarity(sentence1, sentence2, stop_words=None):
    if stop_words is None:
        stop_words = []

    # Lowercase
    sentence1 = [w.lower() for w in sentence1]
    sentence2 = [w.lower() for w in sentence2]

    # Unique words
    all_words = list(set(sentence1 + sentence2))

    # Vector banate hain
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # Sentence1 words count
    for w in sentence1:
        if w in stop_words:
            continue
        vector1[all_words.index(w)] += 1

    # Sentence2 words count
    for w in sentence2:
        if w in stop_words:
            continue
        vector2[all_words.index(w)] += 1

    # Cosine distance se similarity (1 - distance)
    dist = cosine_distance(vector1, vector2)
    if np.isnan(dist):
        return 0

    return 1 - dist


# --------- Similarity matrix banana ---------
def gen_sim_matrix(sentences, stop_words):
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            similarity_matrix[i][j] = sentence_similarity(
                sentences[i], sentences[j], stop_words
            )

    return similarity_matrix


# --------- Summary generate karna ---------
def generate_summary(file_name, top_n=5):
    # Stopwords ensure
    try:
        stop_words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        stop_words = stopwords.words("english")

    # File se sentences
    sentences = read_article(file_name)
    if len(sentences) == 0:
        print("No sentences found in file.")
        return

    # Similarity matrix
    sim_matrix = gen_sim_matrix(sentences, stop_words)

    # Graph + PageRank
    sentence_similarity_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Sentences ko score ke hisaab se sort karo
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True,
    )

    # Top N sentences summary
    summary_sentences = []
    for i in range(min(top_n, len(ranked_sentences))):
        summary_sentences.append(" ".join(ranked_sentences[i][1]))

    print("Summary:\n")
    print(". ".join(summary_sentences))


# --------- Direct run karne ke liye example ---------
if __name__ == "__main__":
    # yahan apni text file ka naam daalo
    generate_summary("article.txt", top_n=5)
