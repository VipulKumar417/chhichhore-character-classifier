"""
Universal Script Parser Module
Extracts character dialogues from movie scripts in various formats.
"""

import re
from typing import Dict, List, Tuple
from collections import defaultdict


class UniversalScriptParser:
    """
    Parse movie scripts from different formats to extract character dialogues.
    """
    
    def __init__(self):
        # Common character name patterns
        self.char_patterns = [
            # Standard screenplay: CHARACTER NAME in caps at start of line
            r'^[\s]*([A-Z][A-Z\s\.\-\']+)(?:\s*\([^)]*\))?[\s]*$',
            # Character with (V.O.) or (O.S.)
            r'^[\s]*([A-Z][A-Z\s\.\-\']+)\s*\((V\.O\.|O\.S\.|CONT\'D|CONTD)\)',
            # GitHub dialogue format: CHARACTER=>DIALOGUE
            r'^([A-Z][A-Za-z\s\.\-\']+)\s*=>\s*(.+)$'
        ]
        
        # Words that look like character names but aren't
        self.exclude_names = {
            'INT', 'EXT', 'INTERIOR', 'EXTERIOR', 'CUT TO', 'FADE IN', 'FADE OUT',
            'DISSOLVE TO', 'THE END', 'CONTINUED', 'CONTINUOUS', 'LATER', 'DAY',
            'NIGHT', 'MORNING', 'EVENING', 'ANGLE ON', 'CLOSE ON', 'WIDE SHOT',
            'MEDIUM SHOT', 'POV', 'INSERT', 'FLASHBACK', 'MONTAGE', 'SUPER',
            'TITLE', 'CREDITS', 'END CREDITS', 'BLACK', 'WHITE', 'SCENE',
            'ACT ONE', 'ACT TWO', 'ACT THREE', 'PROLOGUE', 'EPILOGUE'
        }
        
        # Minimum dialogue length to consider
        self.min_dialogue_length = 10
        self.min_character_lines = 5  # Minimum lines for a character to be included
    
    def parse_script(self, script_text: str, 
                     script_format: str = 'auto') -> Dict[str, List[str]]:
        """
        Parse a movie script and extract character dialogues.
        
        Args:
            script_text: Raw script text
            script_format: 'auto', 'screenplay', 'dialogue', or 'raw'
            
        Returns:
            Dictionary mapping character names to list of their dialogues
        """
        if not script_text or len(script_text) < 100:
            return {}
        
        # Detect format if auto
        if script_format == 'auto':
            script_format = self._detect_format(script_text)
        
        if script_format == 'dialogue':
            return self._parse_dialogue_format(script_text)
        else:
            return self._parse_screenplay_format(script_text)
    
    def _detect_format(self, script_text: str) -> str:
        """Detect the script format."""
        # Check for => format (GitHub dialogue format)
        if script_text.count('=>') > 10:
            return 'dialogue'
        return 'screenplay'
    
    def _parse_dialogue_format(self, script_text: str) -> Dict[str, List[str]]:
        """Parse CHARACTER=>DIALOGUE format (from Movie-Script-Database)."""
        character_dialogues = defaultdict(list)
        
        lines = script_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if '=>' not in line:
                continue
            
            parts = line.split('=>', 1)
            if len(parts) != 2:
                continue
            
            character = parts[0].strip().upper()
            dialogue = parts[1].strip()
            
            # Clean character name
            character = self._clean_character_name(character)
            
            if character and dialogue and len(dialogue) >= self.min_dialogue_length:
                if character not in self.exclude_names:
                    character_dialogues[character].append(dialogue)
        
        # Filter characters with too few lines
        return self._filter_minor_characters(character_dialogues)
    
    def _parse_screenplay_format(self, script_text: str) -> Dict[str, List[str]]:
        """Parse standard screenplay format."""
        character_dialogues = defaultdict(list)
        
        lines = script_text.split('\n')
        current_character = None
        current_dialogue = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                # Save current dialogue on blank line
                if current_character and current_dialogue:
                    dialogue = ' '.join(current_dialogue).strip()
                    if len(dialogue) >= self.min_dialogue_length:
                        character_dialogues[current_character].append(dialogue)
                    current_dialogue = []
                continue
            
            # Check if this is a character name
            is_char_name = False
            for pattern in self.char_patterns[:2]:  # Skip dialogue pattern
                match = re.match(pattern, line_stripped)
                if match:
                    potential_char = match.group(1).strip().upper()
                    potential_char = self._clean_character_name(potential_char)
                    
                    if potential_char and potential_char not in self.exclude_names:
                        # Save previous dialogue
                        if current_character and current_dialogue:
                            dialogue = ' '.join(current_dialogue).strip()
                            if len(dialogue) >= self.min_dialogue_length:
                                character_dialogues[current_character].append(dialogue)
                        
                        current_character = potential_char
                        current_dialogue = []
                        is_char_name = True
                        break
            
            if not is_char_name and current_character:
                # This is dialogue content
                cleaned = self._clean_dialogue(line_stripped)
                if cleaned:
                    current_dialogue.append(cleaned)
        
        # Save last dialogue
        if current_character and current_dialogue:
            dialogue = ' '.join(current_dialogue).strip()
            if len(dialogue) >= self.min_dialogue_length:
                character_dialogues[current_character].append(dialogue)
        
        return self._filter_minor_characters(character_dialogues)
    
    def _clean_character_name(self, name: str) -> str:
        """Clean and normalize character names."""
        # Remove parentheticals
        name = re.sub(r'\([^)]*\)', '', name)
        # Remove numbers
        name = re.sub(r'\d+', '', name)
        # Clean whitespace
        name = ' '.join(name.split())
        # Remove single characters
        if len(name) <= 2:
            return ''
        return name.strip()
    
    def _clean_dialogue(self, text: str) -> str:
        """Clean dialogue text."""
        # Remove stage directions in parentheses
        text = re.sub(r'\([^)]*\)', '', text)
        # Remove scene headers
        if text.startswith('INT.') or text.startswith('EXT.'):
            return ''
        # Remove continuation markers
        text = re.sub(r"\(CONT'D\)", '', text)
        text = re.sub(r'\(V\.O\.\)', '', text)
        text = re.sub(r'\(O\.S\.\)', '', text)
        # Clean whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _filter_minor_characters(self, 
                                  character_dialogues: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Filter out characters with too few lines."""
        return {
            char: dialogues 
            for char, dialogues in character_dialogues.items()
            if len(dialogues) >= self.min_character_lines
        }
    
    def get_character_stats(self, 
                           character_dialogues: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Get statistics for each character."""
        stats = {}
        for char, dialogues in character_dialogues.items():
            total_words = sum(len(d.split()) for d in dialogues)
            avg_words = total_words / len(dialogues) if dialogues else 0
            
            stats[char] = {
                'line_count': len(dialogues),
                'total_words': total_words,
                'avg_words_per_line': round(avg_words, 1),
                'sample': dialogues[0][:100] if dialogues else ''
            }
        
        return stats


def parse_script(script_text: str, script_format: str = 'auto') -> Dict[str, List[str]]:
    """
    Convenience function to parse a script.
    
    Args:
        script_text: Raw script text
        script_format: 'auto', 'screenplay', or 'dialogue'
        
    Returns:
        Dictionary mapping character names to dialogues
    """
    parser = UniversalScriptParser()
    return parser.parse_script(script_text, script_format)


def get_training_data_from_scripts(scripts: List[Dict]) -> List[Tuple[str, str, str]]:
    """
    Extract training data from multiple scripts.
    
    Args:
        scripts: List of script dictionaries with 'script_text' and 'movie_name'
        
    Returns:
        List of (dialogue, character_name, movie_name) tuples
    """
    parser = UniversalScriptParser()
    training_data = []
    
    for script in scripts:
        script_text = script.get('script_text', '')
        movie_name = script.get('movie_name', 'Unknown')
        script_format = script.get('format', 'auto')
        
        try:
            character_dialogues = parser.parse_script(script_text, script_format)
            
            for character, dialogues in character_dialogues.items():
                for dialogue in dialogues:
                    training_data.append((dialogue, character, movie_name))
                    
        except Exception as e:
            continue
    
    return training_data


if __name__ == "__main__":
    # Test with sample screenplay format
    sample_screenplay = """
    INT. COFFEE SHOP - DAY
    
    JOHN
    (nervously)
    I really need to tell you something important.
    It's been on my mind for weeks.
    
    SARAH
    What is it? You're scaring me.
    
    JOHN
    I think we should start our own business.
    
    SARAH
    (laughing)
    That's it? I thought you were going to say
    something terrible!
    """
    
    # Test with dialogue format
    sample_dialogue = """
    JOHN => I really need to tell you something important.
    SARAH => What is it? You're scaring me.
    JOHN => I think we should start our own business.
    SARAH => That's it? I thought you were going to say something terrible!
    JOHN => I'm serious! We could be millionaires.
    SARAH => Okay, tell me more about this idea.
    """
    
    print("=" * 50)
    print("UNIVERSAL SCRIPT PARSER TEST")
    print("=" * 50)
    
    parser = UniversalScriptParser()
    parser.min_character_lines = 1  # Lower for testing
    
    print("\n--- Screenplay Format ---")
    result1 = parser.parse_script(sample_screenplay, 'screenplay')
    for char, dialogues in result1.items():
        print(f"{char}: {len(dialogues)} lines")
        for d in dialogues[:2]:
            print(f"  - {d[:50]}...")
    
    print("\n--- Dialogue Format ---")
    result2 = parser.parse_script(sample_dialogue, 'dialogue')
    for char, dialogues in result2.items():
        print(f"{char}: {len(dialogues)} lines")
        for d in dialogues[:2]:
            print(f"  - {d[:50]}...")
