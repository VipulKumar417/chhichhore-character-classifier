"""
Character Archetypes Module
Defines personality archetypes based on common movie character types.
"""

from typing import Dict, List, Tuple
import re


# Character Archetypes with keywords and traits
ARCHETYPES = {
    'THE_LEADER': {
        'name': 'The Leader',
        'description': 'Takes charge, responsible, protective, makes decisions for the group.',
        'traits': ['Responsible', 'Decisive', 'Protective', 'Confident', 'Motivating'],
        'chhichhore_mapping': 'ANNI',
        'keywords': [
            'lead', 'team', 'together', 'plan', 'strategy', 'everyone', 'fight',
            'win', 'never give up', 'responsibility', 'captain', 'chief', 'boss',
            'follow', 'trust', 'believe', 'mission', 'goal', 'unite', 'group',
            'organize', 'manage', 'decide', 'meeting', 'project', 'deadline',
            'sab log', 'chalo', 'karenge', 'hostel', 'competition'
        ],
        'dialogue_patterns': [
            r'we (can|will|must|should)',
            r'follow me',
            r'trust (me|us)',
            r"let's (go|do|fight|win)",
            r'everyone',
            r'together',
            r'sab (log|ko|milke)'
        ]
    },
    
    'THE_ROMANTIC': {
        'name': 'The Romantic',
        'description': 'Focused on love and relationships, emotional about romance.',
        'traits': ['Romantic', 'Emotional', 'Charming', 'Flirty', 'Passionate'],
        'chhichhore_mapping': 'SEXA',
        'keywords': [
            'love', 'heart', 'girl', 'beautiful', 'date', 'marry', 'forever',
            'kiss', 'relationship', 'feelings', 'romance', 'crush', 'attraction',
            'girlfriend', 'boyfriend', 'darling', 'sweetheart', 'soulmate',
            'ladki', 'patana', 'propose', 'pyaar', 'dil', 'ishq', 'mohabbat',
            'hot', 'cute', 'pretty', 'handsome', 'sexy', 'flirt', 'impress',
            'number', 'insta', 'dm', 'slide', 'approach', 'bandi', 'gf', 'bf'
        ],
        'dialogue_patterns': [
            r'i love',
            r'beautiful',
            r'my (heart|love|crush)',
            r'in love',
            r'marry',
            r'(girl|lady|woman|ladki|bandi)',
            r'(hot|cute|pretty)'
        ]
    },
    
    'THE_REBEL': {
        'name': 'The Rebel',
        'description': 'Cool, breaks rules, confident, does things their own way.',
        'traits': ['Cool', 'Rebellious', 'Confident', 'Bold', 'Independent'],
        'chhichhore_mapping': 'DEREK',
        'keywords': [
            'rules', 'break', 'rebel', 'different', 'damn', 'hell', 'fuck',
            'shit', 'whatever', 'cool', 'awesome', 'badass', 'attitude', 'style',
            'swagger', 'chill', 'dgaf', 'idgaf', 'boring', 'system', 'nobody',
            'alone', 'savage', 'sick', 'dope', 'lit', 'fire', 'bro',
            'aukaat', 'bakwas', 'chod', 'apna', 'mera', 'akela'
        ],
        'dialogue_patterns': [
            r"don't (care|give)",
            r'my (way|rules|style)',
            r'screw (this|that|it)',
            r'whatever',
            r'(boring|stupid)',
            r'(fuck|shit|damn)',
            r'idgaf'
        ]
    },
    
    'THE_CAREGIVER': {
        'name': 'The Caregiver',
        'description': 'Nurturing, supportive, always there for others, emotional support.',
        'traits': ['Caring', 'Supportive', 'Empathetic', 'Nurturing', 'Selfless'],
        'chhichhore_mapping': 'MUMMY',
        'keywords': [
            'worried', 'safe', 'protect', 'comfort', 'always here', 'for you',
            'hurt', 'crying', 'sad', 'upset', 'hugs', 'miss you', 'love you',
            'sending', 'prayers', 'bless', 'hospital', 'doctor', 'medicine',
            'fever', 'sick', 'injured', 'accident', 'emergency', 'careful',
            'rona', 'dard', 'takleef', 'khana', 'neend', 'dawai', 'theek'
        ],
        'dialogue_patterns': [
            r'are you (hurt|sick|injured)',
            r'sending (love|hugs|prayers)',
            r'miss you',
            r'love you (so much|dear)',
            r'(worried|scared) about',
            r'(hospital|doctor|medicine)'
        ]
    },
    
    'THE_INTELLECTUAL': {
        'name': 'The Intellectual',
        'description': 'Smart, analytical, passionate about knowledge, thinks deeply.',
        'traits': ['Intelligent', 'Analytical', 'Curious', 'Passionate', 'Thoughtful'],
        'chhichhore_mapping': 'ACID',
        'keywords': [
            'think', 'know', 'learn', 'understand', 'logic', 'reason', 'science',
            'study', 'book', 'theory', 'fact', 'analysis', 'research', 'idea',
            'interesting', 'curious', 'wonder', 'question', 'explain', 'meaning',
            'philosophy', 'truth', 'mind', 'brain', 'calculate', 'figure'
        ],
        'dialogue_patterns': [
            r'think about',
            r'according to',
            r'logically',
            r'the (truth|fact|theory)',
            r'understand',
            r'(research|study|analysis)'
        ]
    },
    
    'THE_COMIC': {
        'name': 'The Comic',
        'description': 'Funny, laid-back, always cracking jokes, life of the party.',
        'traits': ['Funny', 'Easy-going', 'Playful', 'Witty', 'Entertaining'],
        'chhichhore_mapping': 'BEVDA',
        'keywords': [
            'joke', 'funny', 'laugh', 'haha', 'hahaha', 'hehe', 'kidding', 'crazy',
            'party', 'fun', 'drink', 'drunk', 'celebrate', 'dude', 'bruh',
            'hilarious', 'lol', 'lmao', 'rofl', 'lmfao', 'xd', 'comedy', 'prank',
            'maza', 'masti', 'pagal', 'daaru', 'beer', 'cheers', 'shots',
            'bakchodi', 'timepass', 'chutiapa', 'ghanta', 'bsdk', 'bc', 'mc',
            'wtf', 'tf', 'bruh', 'ded', 'dying', 'dead', 'mood', 'vibe'
        ],
        'dialogue_patterns': [
            r'(haha|hahaha|hehe|lol|lmao|lmfao|rofl)',
            r'(joke|joking|kidding|jk)',
            r'(funny|hilarious|ded|dying|dead)',
            r'(party|celebrate|drink|drunk|shots)',
            r'(bakchodi|masti|pagal)',
            r'(bruh|dude|bro)'
        ]
    },
    
    'THE_RIVAL': {
        'name': 'The Rival',
        'description': 'Competitive, challenging, driven by competition and winning.',
        'traits': ['Competitive', 'Ambitious', 'Determined', 'Challenging', 'Proud'],
        'chhichhore_mapping': 'RAGGIE',
        'keywords': [
            'beat', 'win', 'lose', 'competition', 'challenge', 'better', 'best',
            'defeat', 'victory', 'champion', 'rival', 'enemy', 'opponent',
            'stronger', 'faster', 'prove', 'show', 'fight', 'battle', 'war',
            'versus', 'match', 'game', 'score', 'number one', 'loser', 'winner'
        ],
        'dialogue_patterns': [
            r'beat (you|them|him|her)',
            r"i('ll| will) win",
            r'(better|best) than',
            r'(challenge|compete)',
            r'(loser|winner)',
            r'(prove|show) (you|them)'
        ]
    }
}


def get_archetype_info(archetype_key: str) -> Dict:
    """Get information about an archetype."""
    return ARCHETYPES.get(archetype_key, {})


def get_all_archetypes() -> Dict[str, Dict]:
    """Get all archetype definitions."""
    return ARCHETYPES.copy()


def get_chhichhore_mapping() -> Dict[str, str]:
    """Get mapping from archetype to Chhichhore character."""
    return {key: val['chhichhore_mapping'] for key, val in ARCHETYPES.items()}


def get_archetype_keywords() -> Dict[str, List[str]]:
    """Get keywords for each archetype."""
    return {key: val['keywords'] for key, val in ARCHETYPES.items()}


def score_text_for_archetypes(text: str) -> Dict[str, float]:
    """
    Score a text for each archetype based on keyword matches.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary of archetype scores (0-1)
    """
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    scores = {}
    
    for archetype_key, archetype_info in ARCHETYPES.items():
        keywords = set(archetype_info['keywords'])
        patterns = archetype_info['dialogue_patterns']
        
        # Keyword match score
        matched_keywords = words.intersection(keywords)
        keyword_score = len(matched_keywords) / max(len(keywords), 1)
        
        # Pattern match score
        pattern_matches = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                pattern_matches += 1
        pattern_score = pattern_matches / max(len(patterns), 1)
        
        # Combined score (weighted)
        total_score = (keyword_score * 0.6) + (pattern_score * 0.4)
        scores[archetype_key] = round(total_score, 4)
    
    return scores


def classify_dialogue_to_archetype(dialogue: str) -> Tuple[str, float]:
    """
    Classify a single dialogue to its most likely archetype.
    
    Args:
        dialogue: The dialogue text
        
    Returns:
        Tuple of (archetype_key, confidence_score)
    """
    scores = score_text_for_archetypes(dialogue)
    
    if not scores:
        return 'THE_COMIC', 0.0  # Default fallback
    
    best_archetype = max(scores.keys(), key=lambda k: scores[k])
    return best_archetype, scores[best_archetype]


def label_character_dialogues(dialogues: List[str]) -> Tuple[str, Dict[str, float]]:
    """
    Label a character based on all their dialogues.
    
    Args:
        dialogues: List of dialogue strings from one character
        
    Returns:
        Tuple of (predicted_archetype, archetype_scores)
    """
    combined_text = ' '.join(dialogues)
    scores = score_text_for_archetypes(combined_text)
    
    best_archetype = max(scores.keys(), key=lambda k: scores[k])
    return best_archetype, scores


if __name__ == "__main__":
    # Test the archetype classifier
    print("=" * 50)
    print("ARCHETYPE CLASSIFIER TEST")
    print("=" * 50)
    
    test_dialogues = [
        ("We can do this together! Everyone follow me, we'll win!", "THE_LEADER"),
        ("She's so beautiful, I think I'm in love with her.", "THE_ROMANTIC"),
        ("I don't care about the rules, I do things my way.", "THE_REBEL"),
        ("Are you okay? Don't worry, I'm here for you.", "THE_CAREGIVER"),
        ("Logically speaking, this theory doesn't make sense.", "THE_INTELLECTUAL"),
        ("Haha dude that's so funny! Let's party tonight!", "THE_COMIC"),
        ("I will beat you! I'm better than everyone here.", "THE_RIVAL")
    ]
    
    print("\n--- Testing Individual Dialogues ---")
    for dialogue, expected in test_dialogues:
        predicted, confidence = classify_dialogue_to_archetype(dialogue)
        status = "✓" if predicted == expected else "✗"
        print(f"\n{status} Expected: {expected}")
        print(f"  Predicted: {predicted} (confidence: {confidence:.2%})")
        print(f"  Text: \"{dialogue[:50]}...\"")
    
    print("\n--- Archetype Definitions ---")
    for key, info in ARCHETYPES.items():
        print(f"\n{info['name']} → {info['chhichhore_mapping']}")
        print(f"  {info['description'][:60]}...")
