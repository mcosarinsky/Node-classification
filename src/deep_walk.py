from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np

class RandomWalker:
    def __init__(self, graph, walk_length, seed=42):
        self.graph = graph
        self.walk_length = walk_length
        self.seed = seed

    def random_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            neighbors = np.where(self.graph[current_node])[0]
            if len(neighbors) > 0:
                next_node = np.random.choice(neighbors)
                walk.append(next_node)
            else:
                break
        return walk

    def generate_walks(self, num_walks):
        np.random.seed(self.seed)
        nodes = np.arange(len(self.graph))
        walks = []
        for _ in tqdm(range(num_walks)):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(node))
        return walks


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, sentences=None, seed=42):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.sentences = sentences
        self.seed = seed
        self.w2v_model = None
        self.embeddings = None

    def train(self, embedding_size=128, window_size=5, iters=5, verbose=1, **kwargs):
        if self.sentences is None:
          walker = RandomWalker(self.graph, self.walk_length)
          if verbose: print("Generating sentences...")
          sentences = walker.generate_walks(self.num_walks)
        else:
          sentences = self.sentences

        kwargs["seed"] = self.seed
        kwargs["sentences"] = sentences
        kwargs["min_count"] = 0
        kwargs["vector_size"] = embedding_size
        kwargs["negative"] = 0
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # hierarchical Softmax
        kwargs["window"] = window_size
        kwargs["epochs"] = iters

        if verbose: print("\nTraining skipgram...\n")
        self.w2v_model = Word2Vec(**kwargs)
        if verbose: print("Done!")

        return self.w2v_model

    def get_embeddings(self):
        if self.w2v_model is None:
            raise ValueError("Train the model first")

        n_nodes = len(self.graph)
        embedding_size = self.w2v_model.vector_size
        self.embeddings = np.zeros((n_nodes, embedding_size))

        for node in range(n_nodes):
            self.embeddings[node] = self.w2v_model.wv[node]

        return self.embeddings