"""
Train/Index the Semantic Character Classifier.
"""
import os
import sys

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_model import SemanticCharacterClassifier

def main():
    print("="*60)
    print("INDEXING SCRIPT FOR SEMANTIC SEARCH")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "Chhichhore-script.txt")
    output_path = os.path.join(base_dir, "semantic_model.pkl")
    
    clf = SemanticCharacterClassifier()
    clf.train(script_path)
    
    print(f"\nSaving semantic index to: {output_path}")
    clf.save(output_path)
    
    print("\n" + "="*60)
    print("TESTING SEMANTIC MATCHING")
    print("="*60)
    
    test_phrases = [
        "You idiot, what have you done?",
        "We need to win this championship",
        "She is so beautiful, I'm in love",
        "Relax guys, let's have a drink",
        "Don't worry, everything will be fine",
        "I will destroy you in the game"
    ]
    
    for phrase in test_phrases:
        print(f"\nInput: '{phrase}'")
        res = clf.predict(phrase)
        print(f"  -> {res['character']} (Conf: {res['confidence']:.2%})")
        print(f"  Match: {res['matches'][0]}")

if __name__ == "__main__":
    main()
