#!/usr/bin/env python3
"""
MuseCoco Caption to Attributes Normalizer

Converts natural language captions to structured attribute tokens.
Output format: [genre][mood][tempo][intensity][texture]

Usage:
    python scripts/caption_to_attrs.py \
        --input data/metascore_captions.jsonl \
        --output data/metascore_attributes.jsonl \
        --vocab configs/attribute_vocab.yaml

Example:
    Input: "A cheerful jazz piano piece with upbeat tempo"
    Output: [jazz][cheerful][fast][high][sparse]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

# Default attribute vocabularies with synonyms
DEFAULT_VOCAB = {
    "genre": {
        "jazz": ["jazz", "swing", "bebop", "blues"],
        "classical": ["classical", "orchestral", "symphony", "baroque", "romantic"],
        "rock": ["rock", "metal", "punk", "alternative"],
        "pop": ["pop", "dance", "disco", "electronic"],
        "folk": ["folk", "country", "bluegrass", "acoustic"],
        "latin": ["latin", "salsa", "bossa", "tango"],
        "ambient": ["ambient", "atmospheric", "soundscape", "drone"],
        "other": ["experimental", "avant-garde", "fusion"],
    },
    "mood": {
        "cheerful": ["cheerful", "happy", "joyful", "upbeat", "bright", "energetic"],
        "calm": ["calm", "peaceful", "serene", "tranquil", "relaxing", "gentle"],
        "melancholic": ["melancholic", "sad", "sorrowful", "nostalgic", "wistful"],
        "dramatic": ["dramatic", "intense", "powerful", "epic", "grand"],
        "mysterious": ["mysterious", "enigmatic", "dark", "eerie", "haunting"],
        "playful": ["playful", "whimsical", "light", "fun", "bouncy"],
        "romantic": ["romantic", "tender", "loving", "sweet", "sentimental"],
        "neutral": ["neutral", "moderate", "balanced"],
    },
    "tempo": {
        "very_slow": ["very slow", "grave", "largo", "adagio"],
        "slow": ["slow", "lento", "andante"],
        "moderate": ["moderate", "moderato", "allegretto"],
        "fast": ["fast", "allegro", "vivace", "upbeat"],
        "very_fast": ["very fast", "presto", "prestissimo", "rapid"],
    },
    "intensity": {
        "low": ["soft", "quiet", "gentle", "subtle", "low"],
        "medium": ["medium", "moderate", "balanced"],
        "high": ["loud", "strong", "powerful", "intense", "high"],
    },
    "texture": {
        "sparse": ["sparse", "minimal", "simple", "thin"],
        "moderate": ["moderate", "balanced", "medium"],
        "dense": ["dense", "complex", "rich", "thick", "layered"],
    },
}


class AttributeNormalizer:
    """Converts captions to structured attributes."""

    def __init__(self, vocab: Optional[Dict] = None):
        """
        Initialize normalizer.

        Args:
            vocab: Custom vocabulary dict. Uses DEFAULT_VOCAB if None.
        """
        self.vocab = vocab or DEFAULT_VOCAB
        self._build_synonym_map()

    def _build_synonym_map(self) -> None:
        """Build reverse mapping from synonym to canonical attribute."""
        self.synonym_map: Dict[str, Tuple[str, str]] = {}
        for attr_type, categories in self.vocab.items():
            for canonical, synonyms in categories.items():
                for synonym in synonyms:
                    # Lowercase and strip for matching
                    key = synonym.lower().strip()
                    self.synonym_map[key] = (attr_type, canonical)

    def normalize(self, caption: str) -> Dict[str, str]:
        """
        Convert caption to attribute dict.

        Args:
            caption: Natural language description

        Returns:
            Dict with keys: genre, mood, tempo, intensity, texture
        """
        caption_lower = caption.lower()
        attributes = {}

        # Extract each attribute type
        for attr_type in ["genre", "mood", "tempo", "intensity", "texture"]:
            attributes[attr_type] = self._extract_attribute(caption_lower, attr_type)

        return attributes

    def _extract_attribute(self, caption: str, attr_type: str) -> str:
        """
        Extract single attribute from caption.

        Args:
            caption: Lowercased caption
            attr_type: One of genre, mood, tempo, intensity, texture

        Returns:
            Canonical attribute value or "unknown"
        """
        # Multi-word phrases first (e.g., "very slow" before "slow")
        candidates = []
        for synonym, (atype, canonical) in self.synonym_map.items():
            if atype != attr_type:
                continue
            # Word boundary matching
            pattern = r"\b" + re.escape(synonym) + r"\b"
            if re.search(pattern, caption):
                candidates.append((len(synonym), canonical))

        if candidates:
            # Prefer longest match (more specific)
            candidates.sort(reverse=True, key=lambda x: x[0])
            return candidates[0][1]

        return "unknown"

    def to_token_string(self, attributes: Dict[str, str]) -> str:
        """
        Convert attribute dict to token string.

        Args:
            attributes: Dict with genre, mood, tempo, intensity, texture

        Returns:
            String like "[jazz][cheerful][fast][high][sparse]"
        """
        order = ["genre", "mood", "tempo", "intensity", "texture"]
        tokens = [f"[{attributes.get(k, 'unknown')}]" for k in order]
        return "".join(tokens)


def load_vocab(path: Path) -> Dict:
    """Load vocabulary from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_captions(
    input_path: Path,
    output_path: Path,
    normalizer: AttributeNormalizer,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Process JSONL file of captions.

    Args:
        input_path: Input JSONL with {loop_id, caption} entries
        output_path: Output JSONL with {loop_id, caption, attributes, tokens}
        normalizer: AttributeNormalizer instance
        verbose: Log each conversion

    Returns:
        (total_count, unknown_count) tuple
    """
    total = 0
    unknown_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            entry = json.loads(line)
            caption = entry.get("caption", "")

            # Normalize
            attributes = normalizer.normalize(caption)
            tokens = normalizer.to_token_string(attributes)

            # Count unknowns
            unknown_attrs = [k for k, v in attributes.items() if v == "unknown"]
            if unknown_attrs:
                unknown_count += 1

            # Write output
            output_entry = {
                "loop_id": entry.get("loop_id"),
                "caption": caption,
                "attributes": attributes,
                "tokens": tokens,
            }
            outfile.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

            if verbose:
                logging.info(
                    f"{entry.get('loop_id')}: {caption[:50]}... -> {tokens}"
                    + (f" (missing: {unknown_attrs})" if unknown_attrs else "")
                )

            total += 1

    return total, unknown_count


def validate_output(output_path: Path) -> Dict[str, int]:
    """
    Validate output file and return statistics.

    Args:
        output_path: Output JSONL file

    Returns:
        Dict with attribute coverage stats
    """
    stats = {
        "total": 0,
        "unknown_genre": 0,
        "unknown_mood": 0,
        "unknown_tempo": 0,
        "unknown_intensity": 0,
        "unknown_texture": 0,
    }

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            attrs = entry.get("attributes", {})
            stats["total"] += 1

            for attr_type in ["genre", "mood", "tempo", "intensity", "texture"]:
                if attrs.get(attr_type) == "unknown":
                    stats[f"unknown_{attr_type}"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert captions to MuseCoco attribute tokens"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with captions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file with attributes",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        help="Custom vocabulary YAML (uses defaults if omitted)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output and print statistics",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log each conversion",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load vocabulary
    if args.vocab:
        logging.info(f"Loading vocabulary from {args.vocab}")
        vocab = load_vocab(args.vocab)
    else:
        logging.info("Using default vocabulary")
        vocab = DEFAULT_VOCAB

    # Initialize normalizer
    normalizer = AttributeNormalizer(vocab)
    logging.info(
        f"Loaded {len(normalizer.synonym_map)} synonyms across "
        f"{len(vocab)} attribute types"
    )

    # Process captions
    logging.info(f"Processing {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total, unknown = process_captions(
        args.input, args.output, normalizer, verbose=args.verbose
    )

    logging.info(f"Processed {total} captions -> {args.output}")
    logging.info(f"Entries with unknown attributes: {unknown} ({unknown/total*100:.1f}%)")

    # Validate if requested
    if args.validate:
        logging.info("Validating output...")
        stats = validate_output(args.output)
        logging.info("Validation stats:")
        for key, value in stats.items():
            if key != "total":
                pct = value / stats["total"] * 100 if stats["total"] > 0 else 0
                logging.info(f"  {key}: {value} ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
