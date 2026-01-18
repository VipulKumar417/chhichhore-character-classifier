"""
WhatsApp Chat Parser Module
Parses WhatsApp chat exports and aggregates messages per user.
"""

import re
from typing import Dict, List, Tuple
from collections import defaultdict


def parse_whatsapp_chat(file_path: str, debug: bool = False) -> Dict[str, str]:
    """
    Parse a WhatsApp chat export file and aggregate messages per user.
    
    Args:
        file_path: Path to the WhatsApp chat .txt file
        debug: If True, print debug information
        
    Returns:
        Dictionary mapping user names to their combined messages
    """
    user_messages = defaultdict(list)
    
    # Supports multiple WhatsApp export formats:
    # 1) DD/MM/YY, HH:MM - Sender: Message
    # 2) DD/MM/YYYY, HH:MM am/pm - Sender: Message
    # 3) [DD/MM/YY, HH:MM:SS] Sender: Message (iOS format)
    # 4) M/D/YY, HH:MM - Sender: Message (US format)
    pattern = (
        r'^'                                   # start of line
        r'(?:\[)?'                             # optional opening bracket (iOS)
        r'(?:(\d{1,2}/\d{1,2}/\d{2,4}),?\s+)?'  # optional date: D/M/YY or DD/MM/YYYY + comma
        r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[aApP][mM])?)\s*'  # time: HH:MM or HH:MM:SS, with optional am/pm
        r'(?:\])?\s*'                          # optional closing bracket (iOS)
        r'-\s*'                                # dash with optional spaces
        r'([^:]+):\s*'                         # sender (anything up to colon)
        r'(.+)$'                               # message
    )
    
    # Read file with various encodings
    lines = []
    for encoding in ['utf-8-sig', 'utf-8', 'utf-16', 'latin-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            if lines:
                if debug:
                    print(f"  Read file with {encoding} encoding, {len(lines)} lines")
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if not lines:
        if debug:
            print("  Error: Could not read file with any encoding")
        return {}
    
    current_user = None
    current_message = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        
        if match:
            # Save previous message if exists
            if current_user and current_message:
                user_messages[current_user].append(current_message)
            
            # Groups: 1=date, 2=time, 3=sender, 4=message
            sender = match.group(3).strip()
            message = match.group(4).strip()
            
            # Skip system messages
            if _is_system_message(sender, message):
                current_user = None
                current_message = None
                continue
            
            # Skip media omitted messages
            if '<Media omitted>' in message or 'This message was deleted' in message:
                current_user = None
                current_message = None
                continue
            
            current_user = _normalize_user_name(sender)
            current_message = message
        else:
            # Multi-line message continuation
            if current_user and current_message:
                current_message += ' ' + line
    
    # Save last message
    if current_user and current_message:
        user_messages[current_user].append(current_message)
    
    # Aggregate messages per user
    aggregated = {}
    for user, messages in user_messages.items():
        aggregated[user] = ' '.join(messages)
    
    return aggregated


def _is_system_message(sender: str, message: str) -> bool:
    """Check if message is a system message."""
    system_indicators = [
        'created group',
        'added',
        'left',
        'removed',
        'changed the subject',
        'changed this group',
        'Messages and calls are end-to-end encrypted',
        'POLL:',
        'OPTION:',
        'joined using a group link',
        'was added',
        'security code changed',
        'changed the group description',
        'changed the group icon',
        'pinned a message',
        'deleted this message',
    ]
    
    full_text = f"{sender}: {message}"
    return any(indicator.lower() in full_text.lower() for indicator in system_indicators)


def _normalize_user_name(name: str) -> str:
    """Normalize user names by removing special characters and phone numbers."""
    # Remove phone number patterns
    if name.startswith('+91'):
        return name  # Keep phone numbers as-is for now
    
    # Remove special symbols like @, ~, etc.
    name = re.sub(r'[@~⁩⁨]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def get_user_stats(user_messages: Dict[str, str]) -> Dict[str, Dict]:
    """
    Get statistics for each user.
    
    Args:
        user_messages: Dictionary of user -> combined messages
        
    Returns:
        Dictionary with user statistics
    """
    stats = {}
    for user, messages in user_messages.items():
        words = messages.split()
        stats[user] = {
            'total_chars': len(messages),
            'total_words': len(words),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1)
        }
    return stats


if __name__ == "__main__":
    # Test the parser
    import os
    
    chat_file = os.path.join(os.path.dirname(__file__), "chat.txt")
    
    if os.path.exists(chat_file):
        messages = parse_whatsapp_chat(chat_file)
        
        print("=" * 50)
        print("WHATSAPP CHAT PARSER TEST")
        print("=" * 50)
        
        for user, msg in messages.items():
            print(f"\n{user}: {len(msg)} characters")
            print(f"  Preview: {msg[:100]}...")
    else:
        print(f"Chat file not found: {chat_file}")
