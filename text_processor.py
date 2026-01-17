"""
Text Processor Module
NLP preprocessing using NLTK for text normalization.
"""

import re
import string
from typing import List

# Try to import NLTK, handle if not available
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# Hindi stopwords (common words to remove)
HINDI_STOPWORDS = {
    'hai', 'he', 'ho', 'hain', 'tha', 'the', 'thi', 'ka', 'ki', 'ke', 'ko',
    'ne', 'se', 'me', 'pe', 'par', 'mein', 'ke', 'ka', 'ki', 'bhi', 'to',
    'toh', 'ye', 'yeh', 'wo', 'woh', 'kya', 'kaise', 'kyun', 'kyon', 'kaun',
    'ab', 'abhi', 'aur', 'ya', 'nahi', 'nhi', 'na', 'mat', 'haan', 'han',
    'ji', 'sir', 'bhai', 'yaar', 'arre', 'abe', 'oye', 'bc', 'bsdk', 'mc',
    'be', 'bey', 'ek', 'do', 'teen', 'isliye', 'isiliye', 'fir', 'phir',
    'jab', 'tab', 'waha', 'yaha', 'kaha', 'le', 'de', 'kar', 'karo', 'karna',
    'raha', 'rahe', 'rhi', 'gaya', 'gaye', 'gayi', 'aaya', 'aaye', 'aayi',
    'tha', 'the', 'thi', 'hoga', 'hogi', 'honge', 'tera', 'mera', 'uska',
    'tumhara', 'aapka', 'kuch', 'sab', 'bahut', 'thoda', 'bohot', 'boht'
}

# English common words in Hinglish chat
COMMON_CHAT_WORDS = {
    'ok', 'okay', 'lol', 'haha', 'hehe', 'lmao', 'xd', 'omg', 'wtf', 'idk',
    'btw', 'brb', 'gtg', 'dm', 'pm', 'hi', 'hello', 'bye', 'thanks', 'thnx',
    'thx', 'pls', 'plz', 'plss', 'k', 'kk', 'hmm', 'umm', 'ohh', 'ahh',
    'yes', 'no', 'ya', 'nope', 'yep', 'yeah', 'nah'
}


def download_nltk_data():
    """Download required NLTK data."""
    if not NLTK_AVAILABLE:
        print("NLTK not installed. Using basic preprocessing.")
        return
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)


def preprocess_text(text: str, 
                    lowercase: bool = True,
                    remove_stopwords: bool = True,
                    remove_punctuation: bool = True,
                    remove_emojis: bool = True,
                    remove_urls: bool = True,
                    remove_mentions: bool = True) -> str:
    """
    Preprocess text for NLP analysis.
    
    Args:
        text: Input text to process
        lowercase: Convert to lowercase
        remove_stopwords: Remove common stopwords
        remove_punctuation: Remove punctuation marks
        remove_emojis: Remove emoji characters
        remove_urls: Remove URLs
        remove_mentions: Remove @mentions
        
    Returns:
        Preprocessed text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions (@username)
    if remove_mentions:
        text = re.sub(r'@[^\s@]+', '', text)
    
    # Remove emojis (Unicode emoji ranges)
    if remove_emojis:
        text = remove_emoji(text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
    else:
        tokens = text.split()
    
    # Remove stopwords
    if remove_stopwords:
        if NLTK_AVAILABLE:
            try:
                english_stopwords = set(stopwords.words('english'))
            except:
                english_stopwords = set()
        else:
            english_stopwords = set()
        
        all_stopwords = english_stopwords | HINDI_STOPWORDS | COMMON_CHAT_WORDS
        tokens = [t for t in tokens if t.lower() not in all_stopwords]
    
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    
    # Remove numbers/digits only tokens
    tokens = [t for t in tokens if not t.isdigit()]
    
    return ' '.join(tokens)


def remove_emoji(text: str) -> str:
    """Remove emoji characters from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    if NLTK_AVAILABLE:
        try:
            return word_tokenize(text.lower())
        except:
            return text.lower().split()
    return text.lower().split()


def extract_features(text: str) -> dict:
    """
    Extract stylistic features from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of extracted features
    """
    words = text.split()
    
    features = {
        'word_count': len(words),
        'char_count': len(text),
        'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'question_marks': text.count('?'),
        'exclamations': text.count('!'),
        'emoji_count': len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', text)),
        'laugh_count': len(re.findall(r'(haha|hehe|lol|lmao|😂|🤣)', text.lower()))
    }
    
    return features


if __name__ == "__main__":
    # Test the preprocessor
    test_texts = [
        "Bhai kya kar raha hai 😂😂 lol",
        "Happy birthday bhai! 🎉🎉 Party kab?",
        "https://example.com check this @⁨Krishna Allen⁩",
        "Main soch raha tha ki poori raat quantum physics chaat te hue kaise kategi."
    ]
    
    print("=" * 50)
    print("TEXT PROCESSOR TEST")
    print("=" * 50)
    
    download_nltk_data()
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Processed: {preprocess_text(text)}")
