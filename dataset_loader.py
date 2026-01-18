"""
Dataset Loader Module
Downloads and loads movie script datasets from Hugging Face and GitHub.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Try to import pandas and datasets
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("pandas not available. Install with: pip install pandas")

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("HuggingFace datasets not available. Install with: pip install datasets")


# Configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "movie_scripts")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Aveek-Saha/Movie-Script-Database/master"


class DatasetLoader:
    """
    Load movie scripts from multiple sources.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded scripts
        """
        self.cache_dir = cache_dir or CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.datasets = {
            'ismaelmousa_movies': {
                'name': 'IsmaelMousa/movies',
                'type': 'huggingface',
                'format': 'json',
                'columns': {'title': 'Name', 'script': 'Script'}
            },
            'imsdb_comedy': {
                'name': 'aneeshas/imsdb-comedy-movie-scripts',
                'type': 'huggingface',
                'format': 'parquet',
                'columns': {'title': 'title', 'script': 'script'}
            },
            'imsdb_action': {
                'name': 'aneeshas/imsdb-500tokenaction-movie-scripts',
                'type': 'huggingface',
                'format': 'parquet',
                'columns': {'title': 'title', 'script': 'script'}
            }
        }
    
    def load_huggingface_dataset(self, dataset_key: str) -> List[Dict[str, str]]:
        """
        Load a dataset from Hugging Face.
        
        Args:
            dataset_key: Key from self.datasets
            
        Returns:
            List of {movie_name, script_text} dictionaries
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets library required. Install: pip install datasets")
        
        config = self.datasets.get(dataset_key)
        if not config:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        print(f"Loading {config['name']} from Hugging Face...")
        
        try:
            dataset = load_dataset(config['name'], split='train')
            
            title_col = config['columns']['title']
            script_col = config['columns']['script']
            
            scripts = []
            for item in dataset:
                title = item.get(title_col, 'Unknown')
                script = item.get(script_col, '')
                
                if script and len(script) > 100:  # Filter very short scripts
                    scripts.append({
                        'movie_name': title,
                        'script_text': script,
                        'source': config['name']
                    })
            
            print(f"  Loaded {len(scripts)} scripts from {config['name']}")
            return scripts
            
        except Exception as e:
            print(f"  Error loading {config['name']}: {e}")
            return []
    
    def load_github_scripts(self, max_scripts: int = 500) -> List[Dict[str, str]]:
        """
        Load scripts from Aveek-Saha/Movie-Script-Database GitHub repo.
        Uses the pre-parsed dialogue files.
        
        Args:
            max_scripts: Maximum number of scripts to download
            
        Returns:
            List of {movie_name, script_text} dictionaries
        """
        print("Loading scripts from GitHub Movie-Script-Database...")
        
        # First, get the metadata to know which files exist
        metadata_url = f"{GITHUB_RAW_BASE}/scripts/metadata/meta.json"
        
        try:
            response = requests.get(metadata_url, timeout=30)
            if response.status_code != 200:
                print(f"  Could not fetch metadata: HTTP {response.status_code}")
                # Fallback: try to load from local cache
                return self._load_github_from_cache()
            
            metadata = response.json()
            
        except Exception as e:
            print(f"  Error fetching metadata: {e}")
            return self._load_github_from_cache()
        
        scripts = []
        count = 0
        
        for script_id, info in metadata.items():
            if count >= max_scripts:
                break
            
            try:
                # Get the dialogue file
                parsed_info = info.get('parsed', {})
                dialogue_file = parsed_info.get('dialogue', '')
                
                if not dialogue_file:
                    continue
                
                # Check cache first
                cached_path = os.path.join(self.cache_dir, 'github', dialogue_file)
                
                if os.path.exists(cached_path):
                    with open(cached_path, 'r', encoding='utf-8', errors='ignore') as f:
                        script_text = f.read()
                else:
                    # Download the file
                    script_url = f"{GITHUB_RAW_BASE}/scripts/parsed/dialogue/{dialogue_file}"
                    script_response = requests.get(script_url, timeout=10)
                    
                    if script_response.status_code != 200:
                        continue
                    
                    script_text = script_response.text
                    
                    # Cache it
                    os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                    with open(cached_path, 'w', encoding='utf-8') as f:
                        f.write(script_text)
                
                # Get movie name
                movie_name = info.get('tmdb', {}).get('title', '') or \
                            info.get('file', {}).get('name', script_id)
                
                if script_text and len(script_text) > 100:
                    scripts.append({
                        'movie_name': movie_name,
                        'script_text': script_text,
                        'source': 'Movie-Script-Database',
                        'format': 'dialogue'  # Already in CHARACTER=>DIALOGUE format
                    })
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"  Loaded {count} scripts...")
                        
            except Exception as e:
                continue
        
        print(f"  Loaded {len(scripts)} scripts from GitHub")
        return scripts
    
    def _load_github_from_cache(self) -> List[Dict[str, str]]:
        """Load GitHub scripts from local cache if available."""
        cache_path = os.path.join(self.cache_dir, 'github')
        scripts = []
        
        if not os.path.exists(cache_path):
            return scripts
        
        for filename in os.listdir(cache_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(cache_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        script_text = f.read()
                    
                    if script_text and len(script_text) > 100:
                        scripts.append({
                            'movie_name': filename.replace('_dialogue.txt', '').replace('_', ' ').title(),
                            'script_text': script_text,
                            'source': 'Movie-Script-Database (cached)',
                            'format': 'dialogue'
                        })
                except:
                    continue
        
        return scripts
    
    def load_all_datasets(self, include_github: bool = True, 
                          github_max: int = 500) -> List[Dict[str, str]]:
        """
        Load scripts from all available sources.
        
        Args:
            include_github: Whether to include GitHub scripts
            github_max: Max scripts to load from GitHub
            
        Returns:
            Combined list of all scripts
        """
        all_scripts = []
        
        # Load HuggingFace datasets
        for key in self.datasets.keys():
            try:
                scripts = self.load_huggingface_dataset(key)
                all_scripts.extend(scripts)
            except Exception as e:
                print(f"  Skipping {key}: {e}")
        
        # Load GitHub scripts
        if include_github:
            try:
                github_scripts = self.load_github_scripts(max_scripts=github_max)
                all_scripts.extend(github_scripts)
            except Exception as e:
                print(f"  Skipping GitHub: {e}")
        
        print(f"\nTotal scripts loaded: {len(all_scripts)}")
        return all_scripts


def load_all_datasets(include_github: bool = True) -> List[Dict[str, str]]:
    """
    Convenience function to load all movie scripts.
    
    Args:
        include_github: Whether to include GitHub scripts
        
    Returns:
        List of script dictionaries
    """
    loader = DatasetLoader()
    return loader.load_all_datasets(include_github=include_github)


if __name__ == "__main__":
    # Test loading
    print("=" * 50)
    print("DATASET LOADER TEST")
    print("=" * 50)
    
    loader = DatasetLoader()
    
    # Test each source
    print("\n--- Testing HuggingFace Datasets ---")
    for key in loader.datasets.keys():
        try:
            scripts = loader.load_huggingface_dataset(key)
            if scripts:
                print(f"  Sample from {key}: {scripts[0]['movie_name']}")
        except Exception as e:
            print(f"  {key}: {e}")
    
    print("\n--- Testing GitHub Dataset ---")
    github_scripts = loader.load_github_scripts(max_scripts=5)
    for s in github_scripts[:3]:
        print(f"  {s['movie_name']}: {len(s['script_text'])} chars")
