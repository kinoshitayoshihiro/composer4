#!/usr/bin/env python3
"""
Add family/accent_profile/density metadata to existing patterns

Usage:
    python3 scripts/add_pattern_metadata.py \\
        --input data/patterns/stage2_guitar_v3.pickle \\
        --output data/patterns/stage2_guitar_v3_enhanced.pickle
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import joblib
except ImportError:
    import pickle as joblib


def add_pattern_metadata(input_path: str, output_path: str):
    """Add family/accent_profile/density to patterns"""
    
    # Load existing pickle using joblib (handles sklearn models)
    try:
        data = joblib.load(input_path)
    except Exception as e:
        print(f"Error loading with joblib: {e}")
        # Fallback to pickle
        import pickle
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    
    print(f"Loaded pickle: {data.get('version')}")
    print(f"Patterns: {len(data.get('patterns', {}))}")
    
    patterns = data.get("patterns", {})
    
    # Define metadata for key patterns (ChatGPT's examples)
    pattern_metadata = {
        # STRUM 8th patterns
        "STRUM8_CLOSED_A": {
            "family": "STRUM_8_ROCK",
            "accent_profile": [1,0,1,0,1,0,1,0, 1,0,1,0,1,0,1,0],  # 16th x 16
            "density_ql_per_bar": 8.0,
            "allowed_sections": ["Verse", "Chorus"]
        },
        "STRUM8_OPEN_B": {
            "family": "STRUM_8_ROCK",
            "accent_profile": [1,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0],
            "density_ql_per_bar": 4.0,
            "allowed_sections": ["Chorus", "PreChorus"]
        },
        
        # ARP 16th patterns
        "ARP16_BALANCE_A": {
            "family": "ARP_16_BAL",
            "accent_profile": [1,0,0,1, 0,1,0,0, 1,0,0,1, 0,1,0,0],
            "density_ql_per_bar": 12.0,
            "allowed_sections": ["Verse", "Bridge"]
        },
        
        # Fingerpicking patterns
        "FINGER_ARPEGGIATED": {
            "family": "FINGER_ARP",
            "accent_profile": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
            "density_ql_per_bar": 6.0,
            "allowed_sections": ["Intro", "Verse", "Outro"]
        },
        
        # Power chord patterns
        "POWER_CHORD_RHYTHM": {
            "family": "POWER_ROCK",
            "accent_profile": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            "density_ql_per_bar": 10.0,
            "allowed_sections": ["Chorus", "Bridge"]
        }
    }
    
    # Update patterns
    updated_count = 0
    for pattern_id, meta in pattern_metadata.items():
        if pattern_id in patterns:
            # Merge new metadata into existing pattern
            patterns[pattern_id].update(meta)
            updated_count += 1
            print(f"✓ Updated {pattern_id}")
        else:
            print(f"⚠️  Pattern not found: {pattern_id}")
    
    print(f"\nUpdated {updated_count}/{len(pattern_metadata)} patterns")
    
    # Save updated pickle using joblib
    try:
        joblib.dump(data, output_path)
    except Exception as e:
        print(f"Error saving with joblib: {e}")
        # Fallback to pickle
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    
    print(f"✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Add pattern metadata')
    parser.add_argument('--input', required=True, help='Input pickle file')
    parser.add_argument('--output', required=True, help='Output pickle file')
    
    args = parser.parse_args()
    
    add_pattern_metadata(args.input, args.output)


if __name__ == '__main__':
    main()
