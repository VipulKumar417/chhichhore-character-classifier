"""
Train the archetype model using LMSYS Chat-1M dataset.
This dataset contains 1 million real human-AI conversations.
Run with: python train_with_lmsys.py
"""

import os
import sys
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import login

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CharacterClassifier
from archetypes import ARCHETYPES, score_text_for_archetypes

# Login to HuggingFace (token should already be cached)
try:
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
except:
    pass

def extract_human_messages(conversation):
    """Extract only human messages from conversation."""
    messages = []
    for msg in conversation:
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            if content and len(content) > 20:  # Skip very short messages
                messages.append(content)
    return messages


def label_message_by_archetype(text):
    """Score text and return best archetype."""
    scores = score_text_for_archetypes(text)
    if not scores:
        return None, 0
    best = max(scores.keys(), key=lambda k: scores[k])
    return best, scores[best]


def main():
    print("=" * 60)
    print("TRAINING WITH LMSYS CHAT-1M DATASET")
    print("=" * 60)
    
    # Load dataset in streaming mode (too large to download fully)
    print("\n[1/4] Loading LMSYS Chat-1M dataset (streaming)...")
    ds = load_dataset('lmsys/lmsys-chat-1m', split='train', streaming=True)
    
    # Collect samples per archetype
    archetype_samples = defaultdict(list)
    max_samples_per_archetype = 10000
    total_conversations = 50000  # Limit to first 50k conversations
    
    print(f"\n[2/4] Processing {total_conversations} conversations...")
    print("       Extracting human messages and labeling by archetype...")
    
    min_score_threshold = 0.05  # Only include confident classifications
    
    for i, sample in enumerate(ds):
        if i >= total_conversations:
            break
            
        if i % 5000 == 0:
            print(f"  Processed {i}/{total_conversations}...")
            # Check if we have enough samples
            all_full = all(len(samples) >= max_samples_per_archetype 
                          for samples in archetype_samples.values())
            if all_full and len(archetype_samples) == 7:
                print("  All archetypes have enough samples!")
                break
        
        # Only use English conversations
        if sample.get('language') != 'English':
            continue
            
        # Extract human messages
        messages = extract_human_messages(sample.get('conversation', []))
        
        for msg in messages:
            archetype, score = label_message_by_archetype(msg)
            
            if archetype and score >= min_score_threshold:
                if len(archetype_samples[archetype]) < max_samples_per_archetype:
                    archetype_samples[archetype].append(msg)
    
    print("\n[3/4] Preparing training data...")
    for archetype, samples in archetype_samples.items():
        name = ARCHETYPES.get(archetype, {}).get('name', archetype)
        print(f"  {name}: {len(samples)} samples")
    
    # Prepare training data
    texts = []
    labels = []
    
    for archetype, samples in archetype_samples.items():
        for sample in samples:
            texts.append(sample)
            labels.append(archetype)
    
    if len(texts) < 1000:
        print(f"\n⚠️ Warning: Only {len(texts)} samples collected. Results may be poor.")
    
    print(f"\n[4/4] Training classifier with {len(texts)} samples...")
    
    # Create and train classifier
    classifier = CharacterClassifier(model_type='logistic', use_archetypes=True)
    metrics = classifier.train(texts, labels, test_size=0.2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Total samples: {len(texts)}")
    print(f"Classes: {metrics['classes']}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "lmsys_archetype_model.pkl")
    classifier.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Test predictions
    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)
    
    test_texts = [
        "We should all work together on this project!",
        "She's so beautiful, I'm completely in love",
        "I don't care what anyone says, I do my own thing",
        "Are you feeling okay? Let me know if you need anything",
        "Let me explain the logic behind this theory",
        "Haha that's hilarious dude! Let's party!",
        "I will definitely beat you in this competition"
    ]
    
    for text in test_texts:
        result = classifier.predict_archetype(text)
        print(f"\nText: \"{text[:45]}...\"")
        print(f"  Archetype: {result['archetype']}")
        print(f"  Chhichhore: {result['chhichhore_character']}")
        print(f"  Confidence: {result['confidence']:.2%}")
    
    return classifier


if __name__ == "__main__":
    main()
