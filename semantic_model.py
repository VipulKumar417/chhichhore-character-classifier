"""
Semantic Character Classifier using Sentence Transformers.
Matches input text to character dialogues based on semantic similarity (meaning/emotion)
rather than just keywords.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import pickle
from script_parser import parse_script, CHARACTERS

class SemanticCharacterClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.labels = None
        self.sentences = None
        self.character_mapping = CHARACTERS
    
    def load_model(self):
        """Load the Sentence Transformer model."""
        if self.model is None:
            print(f"Loading Semantic Model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

    def train(self, script_path: str):
        """
        Train by encoding all dialogues in the script.
        Actually just indexing the script.
        """
        self.load_model()
        
        # Parse script
        print(f"Parsing script: {script_path}")
        dialogues = parse_script(script_path)
        
        # Prepare training data (split into sentences for better granularity)
        self.sentences = []
        self.labels = []
        
        print("Encoding script dialogues...")
        import re
        for char, text in dialogues.items():
            # Split into roughly sentence-like chunks
            # Using simple split by punctuation
            chunks = re.split(r'[.!?]+', text)
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) > 10:  # Ignore very short fragments
                    self.sentences.append(chunk)
                    self.labels.append(char)
        
        # Compute embeddings
        self.embeddings = self.model.encode(self.sentences, convert_to_tensor=True)
        print(f"Indexed {len(self.sentences)} sentences.")
        
        # Compute prototype embeddings per character (average center)
        # useful for holistic matching?
        # self.prototypes = {}
        # for char in set(self.labels):
        #     indices = [i for i, label in enumerate(self.labels) if label == char]
        #     char_embeddings = self.embeddings[indices]
        #     self.prototypes[char] = torch.mean(char_embeddings, dim=0)

    def predict(self, text: str, top_k: int = 3) -> dict:
        """
        Predict character based on semantic similarity.
        Finds top_k most similar sentences in script.
        """
        if self.model is None or self.embeddings is None:
            raise ValueError("Model not trained/loaded!")
            
        # Encode input
        query_embedding = self.model.encode(text, convert_to_tensor=True)
        
        # Compute cosine similarities
        # util.cos_sim returns a tensor [[score1, score2, ...]]
        cosine_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # Find top k matches
        # torch.topk returns (values, indices)
        top_results = list(zip(cosine_scores, range(len(cosine_scores))))
        top_results.sort(key=lambda x: x[0], reverse=True)
        top_results = top_results[:top_k]
        
        # Aggregate votes
        votes = Counter()
        matched_Sentences = []
        
        total_score = 0
        for score, idx in top_results:
            char = self.labels[idx]
            sentence = self.sentences[idx]
            # weighting vote by similarity score
            votes[char] += float(score)
            matched_Sentences.append(f"{char}: {sentence} ({score:.2f})")
            total_score += float(score)
        
        # Normalize probabilities (soft of)
        probs = {char: score/total_score for char, score in votes.items()}
        
        best_char = votes.most_common(1)[0][0]
        confidence = probs.get(best_char, 0)
        
        return {
            'character': best_char,
            'confidence': confidence,
            'probabilities': probs,
            'matches': matched_Sentences
        }

    def save(self, path: str):
        """Save the indexed embeddings."""
        with open(path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'labels': self.labels,
                'sentences': self.sentences,
                'model_name': self.model_name
            }, f)
            
    def load(self, path: str):
        """Load indexed embeddings."""
        self.load_model() # Ensure model is loaded
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.labels = data['labels']
            self.sentences = data.get('sentences', [])
