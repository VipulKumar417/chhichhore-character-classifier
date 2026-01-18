"""
Character Classification Model Module
Uses TF-IDF and Logistic Regression for text classification.
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available. Please install: pip install scikit-learn")

from text_processor import preprocess_text, download_nltk_data
from script_parser import parse_script, get_training_data, CHARACTERS

# Import new modules for universal archetype training
try:
    from dataset_loader import DatasetLoader, load_all_datasets
    from universal_parser import UniversalScriptParser, get_training_data_from_scripts
    from archetypes import ARCHETYPES, get_chhichhore_mapping, label_character_dialogues
    ARCHETYPE_MODULES_AVAILABLE = True
except ImportError:
    ARCHETYPE_MODULES_AVAILABLE = False


class CharacterClassifier:
    """
    ML model to classify text as a character archetype or Chhichhore character.
    Supports training on both Chhichhore script and universal movie datasets.
    """
    
    def __init__(self, model_type: str = 'logistic', use_archetypes: bool = False):
        """
        Initialize the classifier.
        
        Args:
            model_type: 'logistic' for Logistic Regression, 'svm' for LinearSVC
            use_archetypes: If True, classify into archetypes; if False, use Chhichhore characters
        """
        self.model_type = model_type
        self.use_archetypes = use_archetypes
        self.vectorizer = None
        self.model = None
        self.classes = None
        self.is_trained = False
        self.archetype_to_chhichhore = get_chhichhore_mapping() if ARCHETYPE_MODULES_AVAILABLE else {}
        self.training_stats = {}
    
    def build_pipeline(self):
        """Build the ML pipeline with TF-IDF and classifier."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Classifier
        if self.model_type == 'svm':
            self.model = LinearSVC(
                C=1.0,
                class_weight='balanced',
                max_iter=5000
            )
        else:
            self.model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000
            )
    
    def train(self, texts: List[str], labels: List[str], test_size: float = 0.2) -> Dict:
        """
        Train the classifier on text data.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels (character names)
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        # Preprocess texts
        download_nltk_data()
        processed_texts = [preprocess_text(t) for t in texts]
        
        # Filter empty texts
        valid_data = [(t, l) for t, l in zip(processed_texts, labels) if t.strip()]
        if len(valid_data) < 10:
            raise ValueError("Not enough training data after preprocessing")
        
        processed_texts, labels = zip(*valid_data)
        
        # Build pipeline
        self.build_pipeline()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Train vectorizer and model
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        self.model.fit(X_train_vec, y_train)
        self.classes = list(self.model.classes_)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'classes': self.classes,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Predict character for a text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (predicted_character, probabilities_dict)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Preprocess
        processed = preprocess_text(text)
        if not processed.strip():
            # Return most common class if text is empty after preprocessing
            return self.classes[0], {c: 1.0/len(self.classes) for c in self.classes}
        
        # Vectorize
        X = self.vectorizer.transform([processed])
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            probabilities = {c: float(p) for c, p in zip(self.classes, proba)}
        else:
            # For SVM, use decision function
            decision = self.model.decision_function(X)[0]
            # Normalize to probabilities
            exp_dec = np.exp(decision - np.max(decision))
            proba = exp_dec / exp_dec.sum()
            probabilities = {c: float(p) for c, p in zip(self.classes, proba)}
        
        return prediction, probabilities
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        """Predict characters for multiple texts."""
        return [self.predict(t) for t in texts]
    
    def load(self, path: str):
        """Load a trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.model = data['model']
        self.classes = data['classes']
        self.model_type = data['model_type']
        self.use_archetypes = data.get('use_archetypes', False)
        self.archetype_to_chhichhore = data.get('archetype_mapping', {})
        self.is_trained = True
    
    def save(self, path: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'classes': self.classes,
            'model_type': self.model_type,
            'use_archetypes': self.use_archetypes,
            'archetype_mapping': self.archetype_to_chhichhore
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def train_from_datasets(self, include_github: bool = True, 
                           max_samples_per_archetype: int = 5000,
                           test_size: float = 0.4) -> Dict:
        """
        Train the classifier using movie script datasets.
        Labels characters by their archetype.
        
        Args:
            include_github: Whether to include GitHub scripts
            max_samples_per_archetype: Max training samples per archetype
            test_size: Fraction for testing
            
        Returns:
            Dictionary with training metrics
        """
        if not ARCHETYPE_MODULES_AVAILABLE:
            raise ImportError("Archetype modules not available. Check imports.")
        
        self.use_archetypes = True
        print("=" * 50)
        print("TRAINING FROM MOVIE DATASETS")
        print("=" * 50)
        
        # Load datasets
        print("\n[1/4] Loading movie scripts...")
        loader = DatasetLoader()
        scripts = loader.load_all_datasets(include_github=include_github)
        
        if len(scripts) < 10:
            raise ValueError(f"Not enough scripts loaded: {len(scripts)}")
        
        # Parse scripts to extract dialogues
        print(f"\n[2/4] Parsing {len(scripts)} scripts...")
        parser = UniversalScriptParser()
        
        archetype_dialogues = {key: [] for key in ARCHETYPES.keys()}
        
        for i, script in enumerate(scripts):
            try:
                script_text = script.get('script_text', '')
                script_format = script.get('format', 'auto')
                
                character_dialogues = parser.parse_script(script_text, script_format)
                
                # Label each character by archetype
                for character, dialogues in character_dialogues.items():
                    if len(dialogues) < 5:
                        continue
                    
                    # Classify character to archetype
                    archetype, scores = label_character_dialogues(dialogues)
                    
                    # Add dialogues to that archetype
                    for dialogue in dialogues[:50]:  # Limit per character
                        if len(dialogue) > 20:
                            archetype_dialogues[archetype].append(dialogue)
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(scripts)} scripts...")
                    
            except Exception as e:
                continue
        
        # Prepare training data
        print("\n[3/4] Preparing training data...")
        texts = []
        labels = []
        
        for archetype, dialogues in archetype_dialogues.items():
            # Limit samples per archetype
            sampled = dialogues[:max_samples_per_archetype]
            texts.extend(sampled)
            labels.extend([archetype] * len(sampled))
            print(f"  {ARCHETYPES[archetype]['name']}: {len(sampled)} samples")
        
        if len(texts) < 100:
            raise ValueError(f"Not enough training data: {len(texts)} samples")
        
        # Train
        print(f"\n[4/4] Training classifier with {len(texts)} samples...")
        metrics = self.train(texts, labels, test_size=test_size)
        
        # Store stats
        self.training_stats = {
            'total_scripts': len(scripts),
            'total_samples': len(texts),
            'samples_per_archetype': {
                arch: len(dialogues) for arch, dialogues in archetype_dialogues.items()
            }
        }
        
        print(f"\nTraining complete! Accuracy: {metrics['accuracy']:.2%}")
        return metrics
    
    def predict_archetype(self, text: str) -> Dict:
        """
        Predict archetype with Chhichhore character mapping.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with archetype, chhichhore character, and probabilities
        """
        prediction, probabilities = self.predict(text)
        
        if self.use_archetypes:
            archetype_key = prediction
            archetype_info = ARCHETYPES.get(archetype_key, {})
            chhichhore_char = self.archetype_to_chhichhore.get(archetype_key, 'ANNI')
            
            return {
                'archetype': archetype_info.get('name', prediction),
                'archetype_key': archetype_key,
                'archetype_description': archetype_info.get('description', ''),
                'chhichhore_character': chhichhore_char,
                'chhichhore_info': CHARACTERS.get(chhichhore_char, {}),
                'confidence': probabilities.get(prediction, 0.0),
                'all_probabilities': probabilities
            }
        else:
            # Legacy mode: direct Chhichhore character prediction
            return {
                'chhichhore_character': prediction,
                'chhichhore_info': CHARACTERS.get(prediction, {}),
                'confidence': probabilities.get(prediction, 0.0),
                'all_probabilities': probabilities
            }


def train_from_script(script_path: str, save_path: Optional[str] = None) -> CharacterClassifier:
    """
    Train a classifier using the movie script.
    
    Args:
        script_path: Path to the Chhichhore script file
        save_path: Optional path to save the trained model
        
    Returns:
        Trained CharacterClassifier
    """
    print("Loading training data from script...")
    training_data = get_training_data(script_path)
    
    if len(training_data) < 50:
        print(f"Warning: Only {len(training_data)} training samples found.")
    
    texts, labels = zip(*training_data)
    
    print(f"Training classifier with {len(texts)} samples...")
    classifier = CharacterClassifier(model_type='logistic')
    metrics = classifier.train(list(texts), list(labels))
    
    print(f"Training complete! Accuracy: {metrics['accuracy']:.2%}")
    print(f"Classes: {metrics['classes']}")
    
    if save_path:
        classifier.save(save_path)
        print(f"Model saved to: {save_path}")
    
    return classifier


if __name__ == "__main__":
    # Test the model
    script_file = os.path.join(os.path.dirname(__file__), "Chhichhore-script.txt")
    model_path = os.path.join(os.path.dirname(__file__), "character_model.pkl")
    
    if os.path.exists(script_file):
        print("=" * 50)
        print("CHARACTER CLASSIFIER TEST")
        print("=" * 50)
        
        # Train the model
        classifier = train_from_script(script_file, model_path)
        
        # Test predictions
        test_texts = [
            "Yaar bahut maza aa gaya, phir se chalte hain!",
            "Main bahut emotional ho gaya, mujhe rona aa raha hai",
            "Chal saale, tujhe challenge karta hoon!",
            "Ha bhai, sab theek hai, chill maar"
        ]
        
        print("\n" + "=" * 50)
        print("TEST PREDICTIONS")
        print("=" * 50)
        
        for text in test_texts:
            pred, probs = classifier.predict(text)
            char_info = CHARACTERS.get(pred, {})
            print(f"\nText: {text}")
            print(f"Predicted: {pred} - {char_info.get('description', '')}")
            print(f"Confidence: {probs[pred]:.2%}")
    else:
        print(f"Script file not found: {script_file}")
