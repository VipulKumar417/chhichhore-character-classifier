"""
Measure accuracy of the Hybrid Classifier on Chhichhore script.
"""
import os
import sys
import numpy as np
from collections import Counter
import pickle

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CharacterClassifier
from semantic_model import SemanticCharacterClassifier
from script_parser import parse_script, CHARACTERS

# --------------------------------------------------------------------------
# Hybrid Classifier Definition (Copied from app.py to avoid streamlit deps)
# --------------------------------------------------------------------------
class HybridClassifier:
    """Wrapper that combines predictions from multiple models (Legacy + Archetype + Semantic)."""
    def __init__(self, legacy_clf, archetype_clf, semantic_clf):
        self.legacy = legacy_clf
        self.archetype = archetype_clf
        self.semantic = semantic_clf
        self.use_archetypes = True
        
    def predict_archetype(self, text: str) -> dict:
        legacy_res = self.legacy.predict_archetype(text) if self.legacy else None
        archetype_res = self.archetype.predict_archetype(text) if self.archetype else None
        semantic_res = self.semantic.predict(text) if self.semantic else None
        
        # Collect confidences
        legacy_conf = legacy_res.get('confidence', 0) if legacy_res else 0
        arch_conf = archetype_res.get('confidence', 0) if archetype_res else 0
        
        semantic_conf = 0.0
        if semantic_res:
             semantic_conf = semantic_res.get('confidence', 0)
        
        # Priority Logic (Matches app.py)
        scores = [
            (legacy_conf * 3.0, 'legacy'),
            (semantic_conf * 2.0, 'semantic'),
            (arch_conf * 1.0, 'archetype')
        ]
        
        winner_score, winner_model = max(scores, key=lambda x: x[0])
        
        if winner_model == 'legacy' and legacy_res:
            return legacy_res
        elif winner_model == 'semantic' and semantic_res:
            char = semantic_res['character']
            return {
                'chhichhore_character': char,
                'confidence': semantic_res['confidence'],
                'archetype': "Semantic Match"
            }
        elif archetype_res:
            return archetype_res
            
        return legacy_res or archetype_res

    def predict(self, text):
        res = self.predict_archetype(text)
        return res.get('chhichhore_character'), {}


def load_hybrid_classifier():
    """Load all classifiers."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    legacy_path = os.path.join(base_dir, "character_model.pkl")
    semantic_path = os.path.join(base_dir, "semantic_model.pkl")
    lmsys_path = os.path.join(base_dir, "lmsys_archetype_model.pkl")
    
    classifiers = {}
    
    if os.path.exists(legacy_path):
        c = CharacterClassifier()
        c.load(legacy_path)
        classifiers['legacy'] = c
    else:
        print("Warning: Legacy model not found")

    if os.path.exists(semantic_path):
        from semantic_model import SemanticCharacterClassifier
        c = SemanticCharacterClassifier()
        c.load(semantic_path)
        classifiers['semantic'] = c
    else:
        print("Warning: Semantic model not found")
        
    if os.path.exists(lmsys_path):
        c = CharacterClassifier(use_archetypes=True)
        c.load(lmsys_path)
        classifiers['archetype'] = c
    else:
        print("Warning: LMSYS model not found")
        
    return HybridClassifier(
        classifiers.get('legacy'), 
        classifiers.get('archetype'),
        classifiers.get('semantic')
    )

def main():
    print("="*60)
    print("MEASURING HYBRID MODEL ACCURACY")
    print("="*60)
    
    clf = load_hybrid_classifier()
    
    # Load script for testing
    script_path = os.path.join(os.path.dirname(__file__), "Chhichhore-script.txt")
    dialogues = parse_script(script_path)
    
    # Create test set (sentence level)
    import re
    test_samples = []
    
    for char, text in dialogues.items():
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        for s in sentences:
            s = s.strip()
            if len(s) > 20: # Only meaningful lines
                test_samples.append((s, char))
    
    print(f"Testing on {len(test_samples)} dialogue samples...")
    
    correct = 0
    total = 0
    char_stats = Counter()
    char_correct = Counter()
    
    for text, true_char in test_samples:
        pred_res = clf.predict_archetype(text)
        pred_char = pred_res.get('chhichhore_character')
        
        total += 1
        char_stats[true_char] += 1
        
        if pred_char == true_char:
            correct += 1
            char_correct[true_char] += 1
            
    accuracy = correct / total
    print(f"\nOVERALL ACCURACY: {accuracy:.2%}")
    print(f"({correct}/{total} correct)\n")
    
    print("ACCURACY BY CHARACTER:")
    print("-" * 30)
    for char, count in char_stats.most_common():
        curr_correct = char_correct[char]
        acc = curr_correct / count
        print(f"{char:10} {acc:.2%} ({curr_correct}/{count})")

if __name__ == "__main__":
    main()
