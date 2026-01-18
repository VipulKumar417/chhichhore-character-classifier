"""
Movie Character Classification Web App
Streamlit application to classify WhatsApp chat participants as Chhichhore characters.
"""

import os
import sys
import streamlit as st

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_parser import parse_whatsapp_chat, get_user_stats
from script_parser import CHARACTERS, get_character_info
from model import CharacterClassifier, train_from_script

# Import archetype info if available
try:
    from archetypes import ARCHETYPES, get_chhichhore_mapping
    ARCHETYPES_AVAILABLE = True
except ImportError:
    ARCHETYPES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Chhichhore Character Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #0f3460, #16213e);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #a2d2ff;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    /* Character card */
    .character-card {
        background: linear-gradient(145deg, #1f4068, #162447);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-left: 5px solid #e94560;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .character-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(233,69,96,0.3);
    }
    
    .user-name {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .character-name {
        color: #e94560;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .character-traits {
        background: rgba(233,69,96,0.2);
        padding: 10px 15px;
        border-radius: 10px;
        color: #ffd6e0;
        margin: 10px 0;
    }
    
    .character-desc {
        color: #a2d2ff;
        font-style: italic;
        line-height: 1.6;
    }
    
    .confidence-bar {
        background: #0f3460;
        border-radius: 10px;
        height: 10px;
        margin-top: 10px;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Stats box */
    .stats-box {
        background: linear-gradient(145deg, #0f3460, #16213e);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e94560;
    }
    
    .stats-label {
        color: #a2d2ff;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    
    /* Upload area */
    .upload-area {
        background: linear-gradient(145deg, #1f4068, #162447);
        border: 2px dashed #e94560;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    }
    
    /* Character reference cards */
    .ref-card {
        background: rgba(31, 64, 104, 0.5);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 3px solid #e94560;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None


def load_classifier():
    """Load or train the classifier. Prefer archetype model if available."""
    base_dir = os.path.dirname(__file__)
    archetype_model_path = os.path.join(base_dir, "archetype_model.pkl")
    legacy_model_path = os.path.join(base_dir, "character_model.pkl")
    script_path = os.path.join(base_dir, "Chhichhore-script.txt")
    
    # Try loading archetype model first (trained on 1000+ movies)
    if os.path.exists(archetype_model_path):
        try:
            classifier = CharacterClassifier(use_archetypes=True)
            classifier.load(archetype_model_path)
            st.session_state.model_type = 'archetype'
            return classifier
        except Exception as e:
            pass
    
    # Fallback to legacy model
    if os.path.exists(legacy_model_path):
        try:
            classifier = CharacterClassifier()
            classifier.load(legacy_model_path)
            st.session_state.model_type = 'legacy'
            return classifier
        except:
            pass
    
    # Train from script if no model exists
    if os.path.exists(script_path):
        classifier = train_from_script(script_path, legacy_model_path)
        st.session_state.model_type = 'legacy'
        return classifier
    
    return None


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Chhichhore Character Classifier</h1>
        <p>Upload your WhatsApp group chat to discover which Chhichhore character each participant resembles!</p>
    </div>
    """, unsafe_allow_html=True)


def render_character_card(user_name: str, prediction_data: dict, message_count: int):
    """Render a character prediction card with archetype info."""
    character = prediction_data.get('chhichhore_character', prediction_data.get('character', 'ANNI'))
    confidence = prediction_data.get('confidence', 0) * 100
    archetype = prediction_data.get('archetype', None)
    archetype_desc = prediction_data.get('archetype_description', '')
    
    char_info = CHARACTERS.get(character, {})
    traits_html = ", ".join(char_info.get('traits', []))
    
    # Build archetype section if available
    archetype_section = ""
    if archetype:
        archetype_section = f'''
        <div style="background: rgba(162,210,255,0.2); padding: 10px 15px; border-radius: 10px; margin: 10px 0;">
            <strong style="color: #a2d2ff;">🎭 Archetype:</strong> <span style="color: #fff;">{archetype}</span><br/>
            <span style="color: #a2d2ff; font-size: 0.9em;">{archetype_desc[:100]}...</span>
        </div>
        '''
    
    st.markdown(f"""
    <div class="character-card">
        <div class="user-name">👤 {user_name}</div>
        <div class="character-name">🎬 {character} - {char_info.get('full_name', character)}</div>
        {archetype_section}
        <div class="character-traits">
            <strong>Traits:</strong> {traits_html}
        </div>
        <div class="character-desc">
            {char_info.get('description', 'No description available.')}
        </div>
        <div style="margin-top: 15px; color: #a2d2ff;">
            <strong>Confidence:</strong> {confidence:.1f}% | <strong>Messages analyzed:</strong> {message_count}
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with character reference."""
    with st.sidebar:
        st.markdown("## 🎬 Character Guide")
        st.markdown("---")
        
        for char, info in CHARACTERS.items():
            with st.expander(f"🎭 {char} - {info['full_name']}"):
                st.markdown(f"**Traits:** {', '.join(info['traits'])}")
                st.markdown(f"*{info['description']}*")
        
        st.markdown("---")
        st.markdown("### 📋 How it works")
        st.markdown("""
        1. Export your WhatsApp group chat
        2. Upload the `.txt` file
        3. Our AI analyzes each participant's messages
        4. Get character predictions based on communication style!
        """)
        
        st.markdown("---")
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **NLP**: NLTK, TF-IDF
        - **ML**: Logistic Regression
        - **Training Data**: Chhichhore Movie Script
        """)


def main():
    """Main application."""
    render_header()
    render_sidebar()
    
    # Load classifier
    with st.spinner("🔄 Loading AI model..."):
        if st.session_state.classifier is None:
            st.session_state.classifier = load_classifier()
    
    if st.session_state.classifier is None:
        st.error("❌ Could not load the classifier. Please ensure Chhichhore-script.txt is in the same directory.")
        return
    
    st.success(f\"✅ AI Model loaded successfully! ({st.session_state.get('model_type', 'legacy').title()} Model)\")
    
    # File upload section
    st.markdown("### 📤 Upload WhatsApp Chat")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Drop your WhatsApp chat export here (.txt format)",
            type=['txt'],
            help="Export chat from WhatsApp: Open chat → Menu → More → Export chat → Without media"
        )
    
    with col2:
        use_sample = st.button("📂 Use Sample Chat", use_container_width=True)
    
    # Process chat
    chat_file_path = None
    
    if use_sample:
        chat_file_path = os.path.join(os.path.dirname(__file__), "chat.txt")
        if not os.path.exists(chat_file_path):
            st.error("Sample chat.txt not found in the directory.")
            chat_file_path = None
    elif uploaded_file:
        # Save uploaded file temporarily
        chat_file_path = os.path.join(os.path.dirname(__file__), "temp_chat.txt")
        with open(chat_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
    
    if chat_file_path and os.path.exists(chat_file_path):
        with st.spinner("🔍 Analyzing messages..."):
            try:
                # Parse chat
                user_messages = parse_whatsapp_chat(chat_file_path)
                user_stats = get_user_stats(user_messages)
                
                if not user_messages:
                    st.warning("⚠️ No valid messages found in the chat file.")
                    return
                
                # Make predictions
                predictions = {}
                for user, messages in user_messages.items():
                    if len(messages) > 20:  # Only classify users with enough messages
                        # Use archetype prediction if available, else legacy
                        if hasattr(st.session_state.classifier, 'predict_archetype') and st.session_state.classifier.use_archetypes:
                            result = st.session_state.classifier.predict_archetype(messages)
                            result['message_count'] = user_stats[user]['total_words']
                            predictions[user] = result
                        else:
                            pred, probs = st.session_state.classifier.predict(messages)
                            predictions[user] = {
                                'character': pred,
                                'chhichhore_character': pred,
                                'confidence': probs.get(pred, 0),
                                'probabilities': probs,
                                'message_count': user_stats[user]['total_words']
                            }
                
                st.session_state.predictions = predictions
                
            except Exception as e:
                st.error(f"❌ Error processing chat: {str(e)}")
                return
    
    # Display results
    if st.session_state.predictions:
        st.markdown("---")
        st.markdown("## 🎭 Character Predictions")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stats-box">
                <div class="stats-number">{len(st.session_state.predictions)}</div>
                <div class="stats-label">Participants Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_messages = sum(p['message_count'] for p in st.session_state.predictions.values())
            st.markdown(f"""
            <div class="stats-box">
                <div class="stats-number">{total_messages}</div>
                <div class="stats-label">Total Words Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_chars = len(set(p['character'] for p in st.session_state.predictions.values()))
            st.markdown(f"""
            <div class="stats-box">
                <div class="stats-number">{unique_chars}</div>
                <div class="stats-label">Unique Characters Found</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Character cards in columns
        predictions_list = list(st.session_state.predictions.items())
        
        for i in range(0, len(predictions_list), 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < len(predictions_list):
                    user, data = predictions_list[i + j]
                    with col:
                        render_character_card(
                            user,
                            data,
                            data['message_count']
                        )
        
        # Character distribution
        st.markdown("---")
        st.markdown("### 📊 Character Distribution")
        
        char_counts = {}
        for p in st.session_state.predictions.values():
            char = p.get('chhichhore_character', p.get('character', 'ANNI'))
            char_counts[char] = char_counts.get(char, 0) + 1
        
        import pandas as pd
        df = pd.DataFrame({
            'Character': list(char_counts.keys()),
            'Count': list(char_counts.values())
        })
        
        st.bar_chart(df.set_index('Character'))


if __name__ == "__main__":
    main()
