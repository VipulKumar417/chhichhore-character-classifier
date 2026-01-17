"""
Chhichhore Movie Script Parser Module
Extracts character dialogues from the movie script.
"""

import re
from typing import Dict, List
from collections import defaultdict


# Main characters from Chhichhore with their personality traits
CHARACTERS = {
    'ANNI': {
        'full_name': 'Aniruddh (Anni)',
        'traits': ['Leader', 'Responsible', 'Caring', 'Thoughtful'],
        'description': 'The protagonist - responsible leader who cares deeply for friends and family.',
        'aliases': ['ANIRUDDH', 'ANNI']
    },
    'SEXA': {
        'full_name': 'Sexa',
        'traits': ['Humorous', 'Witty', 'Playful', 'Fun-loving'],
        'description': 'The joker of the group - always cracking jokes and having fun.',
        'aliases': ['SEXA']
    },
    'DEREK': {
        'full_name': 'Derek',
        'traits': ['Cool', 'Rebellious', 'Confident', 'Bold'],
        'description': 'The cool rebel - confident and does things his own way.',
        'aliases': ['DEREK']
    },
    'MUMMY': {
        'full_name': 'Mummy',
        'traits': ['Caring', 'Supportive', 'Sensitive', 'Emotional'],
        'description': 'The caring friend - always there to support and comfort others.',
        'aliases': ['MUMMY']
    },
    'ACID': {
        'full_name': 'Acid',
        'traits': ['Intense', 'Passionate', 'Direct', 'Honest'],
        'description': 'The intense one - passionate about everything and speaks his mind.',
        'aliases': ['ACID']
    },
    'BEVDA': {
        'full_name': 'Bevda',
        'traits': ['Laid-back', 'Casual', 'Easy-going', 'Friendly'],
        'description': 'The chill guy - relaxed and easy-going personality.',
        'aliases': ['BEVDA']
    },
    'RAGGIE': {
        'full_name': 'Raggie',
        'traits': ['Competitive', 'Challenging', 'Rival', 'Determined'],
        'description': 'The rival - competitive and always up for a challenge.',
        'aliases': ['RAGGIE']
    },
    'MAYA': {
        'full_name': 'Maya',
        'traits': ['Caring', 'Emotional', 'Supportive', 'Loving'],
        'description': 'The caring presence - emotional and deeply supportive.',
        'aliases': ['MAYA']
    }
}


def parse_script(file_path: str) -> Dict[str, str]:
    """
    Parse the Chhichhore movie script and extract dialogues per character.
    
    Args:
        file_path: Path to the script .txt file
        
    Returns:
        Dictionary mapping character names to their combined dialogues
    """
    character_dialogues = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    current_character = None
    current_dialogue = []
    
    # Pattern to match character names (all caps, often followed by dialogue)
    char_pattern = r'^[\s]*([A-Z][A-Z\s]+)(?:\s*\(.*\))?$'
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if not line_stripped:
            continue
        
        # Check if this line is a character name
        match = re.match(char_pattern, line_stripped)
        
        if match:
            potential_char = match.group(1).strip()
            
            # Check if it's one of our tracked characters
            matched_char = None
            for char_key, char_info in CHARACTERS.items():
                if potential_char in char_info['aliases'] or potential_char == char_key:
                    matched_char = char_key
                    break
            
            if matched_char:
                # Save previous character's dialogue
                if current_character and current_dialogue:
                    character_dialogues[current_character].append(' '.join(current_dialogue))
                
                current_character = matched_char
                current_dialogue = []
            elif potential_char in ['CHORUS', 'BOTH', 'ALL', 'DOCTOR', 'NURSE', 
                                   'CLERK', 'WARDEN', 'SERVANT', 'COLLEAGUE',
                                   'MOHIT', 'SOORAJ', 'RAGHAV']:
                # Save current and reset
                if current_character and current_dialogue:
                    character_dialogues[current_character].append(' '.join(current_dialogue))
                current_character = None
                current_dialogue = []
        else:
            # This is dialogue content
            if current_character:
                # Clean the line
                cleaned = _clean_dialogue(line_stripped)
                if cleaned:
                    current_dialogue.append(cleaned)
    
    # Save last dialogue
    if current_character and current_dialogue:
        character_dialogues[current_character].append(' '.join(current_dialogue))
    
    # Aggregate dialogues
    aggregated = {}
    for char, dialogues in character_dialogues.items():
        aggregated[char] = ' '.join(dialogues)
    
    return aggregated


def _clean_dialogue(text: str) -> str:
    """Clean dialogue text by removing stage directions and formatting."""
    # Remove content in parentheses (stage directions)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove page numbers and formatting
    text = re.sub(r'^\d+\.$', '', text)
    text = re.sub(r'\f', '', text)
    
    # Remove scene descriptions (usually start with specific patterns)
    if text.startswith('INT.') or text.startswith('EXT.'):
        return ''
    
    # Remove continuation markers
    text = re.sub(r'\(CONT\'D\)', '', text)
    text = re.sub(r'\(V\.O\.\)', '', text)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def get_character_info() -> Dict[str, Dict]:
    """Return character information dictionary."""
    return CHARACTERS.copy()


def get_training_data(script_file: str) -> List[tuple]:
    """
    Get training data as (text, label) pairs.
    
    Args:
        script_file: Path to script file
        
    Returns:
        List of (dialogue_text, character_name) tuples
    """
    dialogues = parse_script(script_file)
    training_data = []
    
    for character, text in dialogues.items():
        # Split into sentences for more training samples
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short fragments
                training_data.append((sentence, character))
    
    return training_data


if __name__ == "__main__":
    # Test the parser
    import os
    
    script_file = os.path.join(os.path.dirname(__file__), "Chhichhore-script.txt")
    
    if os.path.exists(script_file):
        dialogues = parse_script(script_file)
        
        print("=" * 50)
        print("CHHICHHORE SCRIPT PARSER TEST")
        print("=" * 50)
        
        for char, text in dialogues.items():
            info = CHARACTERS.get(char, {})
            print(f"\n{char} ({info.get('full_name', 'Unknown')})")
            print(f"  Traits: {', '.join(info.get('traits', []))}")
            print(f"  Dialogue length: {len(text)} characters")
            print(f"  Preview: {text[:150]}...")
        
        print("\n" + "=" * 50)
        training = get_training_data(script_file)
        print(f"Total training samples: {len(training)}")
    else:
        print(f"Script file not found: {script_file}")
