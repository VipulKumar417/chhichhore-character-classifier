"""
Train the archetype model using movie datasets.
Run with: python train_archetype_model.py
"""

import os
from model import CharacterClassifier

def main():
    print("=" * 60)
    print("UNIVERSAL CHARACTER ARCHETYPE MODEL TRAINING")
    print("=" * 60)
    
    # Create classifier
    classifier = CharacterClassifier(model_type='logistic', use_archetypes=True)
    
    # Train on datasets (this may take a few minutes)
    print("\nTraining classifier on movie datasets...")
    print("This may take 5-10 minutes depending on your internet speed.\n")
    
    try:
        metrics = classifier.train_from_datasets(
            include_github=True,  # Include GitHub scripts
            max_samples_per_archetype=5000,  # Limit per archetype
            test_size=0.2
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Total training samples: {metrics['train_size']}")
        print(f"Classes: {metrics['classes']}")
        
        # Save the model
        model_path = os.path.join(os.path.dirname(__file__), "archetype_model.pkl")
        classifier.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Test predictions
        print("\n" + "=" * 60)
        print("TEST PREDICTIONS")
        print("=" * 60)
        
        test_texts = [
            "We can do this together! Follow me, team!",
            "She's so beautiful, I think I'm in love",
            "I don't care about rules, I do things my way",
            "Are you okay? Don't worry, I'm here for you",
            "Let's think about this logically",
            "Haha that's so funny! Let's party!",
            "I will beat you! I'm the best!"
        ]
        
        for text in test_texts:
            result = classifier.predict_archetype(text)
            print(f"\nText: \"{text[:40]}...\"")
            print(f"  Archetype: {result['archetype']}")
            print(f"  Chhichhore: {result['chhichhore_character']}")
            print(f"  Confidence: {result['confidence']:.2%}")
        
        return classifier
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
