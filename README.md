# mid-term-project
This project is a Text Summarization System built using Python and Natural Language Processing (NLP). Its main goal is to automatically generate a concise summary from a long text file. The project uses several important Python libraries such as NLTK for text processing, NetworkX for implementing the PageRank algorithm, NumPy for matrix operations, and Scipy (indirectly) for mathematical support.

The program first reads an input article from a text file and splits it into individual sentences. Each sentence is cleaned by removing unwanted characters and converting it into a list of meaningful words. After that, the system calculates the similarity score between every pair of sentences using a cosine distance–based method. These similarity scores are stored in a matrix.

Using this matrix, a graph is created where each sentence becomes a node, and edge weights represent similarity. The PageRank algorithm then ranks the importance of each sentence. Finally, the top-ranked sentences are selected and combined to form the final summary.

This project demonstrates essential NLP concepts, graph-based ranking, and automated text processing to produce meaningful summaries efficiently.
