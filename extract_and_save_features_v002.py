import json
import re
import sqlite3
import logging
from typing import Dict, Any
import spacy
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nlp = spacy.load("en_core_web_sm")


def extract_categorical_features(prompt: str) -> Dict[str, str]:
    """Extract categorical visual and stylistic features from the given text prompt.

    Args:
        prompt (str): The input text describing an image or scene.

    Returns:
        Dict[str, str]: A dictionary of extracted categorical features.
    """
    prompt = prompt.lower()

    art_styles = [
        "romanesque", "baroque", "anime", "pixel", "surreal", "realistic", "steampunk",
        "cyberpunk", "impressionist", "expressionist", "gothic", "renaissance", "concept art",
        "modernist", "minimalist", "sci-fi", "fantasy", "digital painting", "photo-realistic"
    ]

    camera_angles = [
        "close-up", "top-down", "side view", "overhead", "isometric", "wide angle",
        "low angle", "high angle", "first person", "third person", "bird's eye", "macro"
    ]

    lighting_styles = [
        "studio lighting", "dramatic lighting", "natural light", "backlit", "soft lighting",
        "volumetric", "sharp focus", "hard light", "ambient light", "rim lighting", "low key", "high key"
    ]

    times_of_day = [
        "sunset", "dawn", "night", "daytime", "noon", "morning", "day", "afternoon",
        "nighttime", "golden hour", "blue hour", "twilight"
    ]

    colors = [
        "red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white",
        "gray", "brown", "cyan", "magenta", "beige", "turquoise", "gold", "silver"
    ]

    color_palettes = [
        "vibrant", "muted", "monochrome", "pastel", "colorful", "sepia", "high contrast",
        "coherent", "analogous", "complementary", "triadic", "warm", "cool"
    ]

    moods = [
        "peaceful", "dramatic", "eerie", "joyful", "melancholic", "awe", "romantic", "tense",
        "dark", "uplifting", "serene", "mysterious"
    ]

    explicit_markers = [
        "highly detailed", "realistic", "intricate", "ultra detailed", "hyper detailed",
        "sharp focus", "octane render", "volumetric", "ray tracing", "8k", "cg render", "photorealistic"
    ]

    composition_mods = [
        "centered", "rule of thirds", "symmetrical", "zoomed in", "wide shot", "sharp focus",
        "asymmetrical", "framed", "diagonal lines", "negative space", "foreground", "background"
    ]

    known_templates = {
        "artstation trending": "artstation_trending",
        "deviantart": "deviantart",
        "pixiv": "pixiv",
        "concept art": "concept_art"
    }

    features = {
        "art_style": next((style for style in art_styles if style in prompt), "unspecified"),
        "camera_angle": next((angle for angle in camera_angles if angle in prompt), "unspecified"),
        "lighting": next((light for light in lighting_styles if light in prompt), "unspecified"),
        "time_of_day": next((time for time in times_of_day if time in prompt), "unspecified"),
        "color": next((color for color in colors if color in prompt), "unspecified"),
        "color_palette": next((palette for palette in color_palettes if palette in prompt), "unspecified"),
        "emotion_mood": next((mood for mood in moods if mood in prompt), "neutral"),
        "explicitness": "explicit" if any(word in prompt for word in explicit_markers) else "implicit",
        "composition_modifiers": "yes" if any(word in prompt for word in composition_mods) else "no",
        "known_template": next((alias for key, alias in known_templates.items() if key in prompt), "freeform")
    }

    logging.debug(f"Extracted categorical features: {features}")
    return features

def count_modifiers_and_entities(prompt: str) -> Dict[str, Any]:
    """Count adjectives and named entities in a prompt using spaCy NLP.

    Args:
        prompt (str): The input text.

    Returns:
        Dict[str, Any]: A dictionary containing counts and lists of adjectives and named entities.
    """
    doc = nlp(prompt)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    named_entities = [ent.text for ent in doc.ents]

    result = {
        'num_adjectives': len(adjectives),
        'num_named_entities': len(named_entities),
        'adjectives': adjectives,
        'named_entities': named_entities,
    }

    logging.debug(f"Modifier/entity counts: {result}")
    return result


def calculate_complexity(prompt: str) -> Dict[str, Any]:
    """Calculate a complexity score for the given prompt based on simple heuristics.

    Args:
        prompt (str): The input text.

    Returns:
        Dict[str, Any]: A dictionary with word, unique word, comma counts and the complexity score.
    """
    words = re.findall(r'\b\w+\b', prompt)
    word_count = len(words)
    unique_word_count = len(set(words))
    comma_count = prompt.count(',')

    complexity_score = (
        0.3 * word_count +
        0.5 * unique_word_count +
        0.2 * comma_count
    )

    result = {
        'word_count': word_count,
        'unique_word_count': unique_word_count,
        'comma_count': comma_count,
        'complexity_score': round(complexity_score, 2)
    }

    logging.debug(f"Complexity results: {result}")
    return result


def create_database(db_path: str) -> None:
    """Create the SQLite database schema.

    Args:
        db_path (str): The path to the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id TEXT PRIMARY KEY,
            word_count INTEGER,
            unique_word_count INTEGER,
            comma_count INTEGER,
            complexity_score REAL,
            num_adjectives INTEGER,
            num_named_entities INTEGER,
            adjectives TEXT,
            named_entities TEXT,
            art_style TEXT,
            camera_angle TEXT,
            lighting TEXT,
            time_of_day TEXT,
            color TEXT,
            color_palette TEXT,
            emotion_mood TEXT,
            explicitness TEXT,
            composition_modifiers TEXT,
            known_template TEXT
        )
    """)
    conn.commit()
    conn.close()
    logging.debug("Database created and schema initialized.")


def insert_feature(db_path: str, key: str, feature: Dict[str, Any]) -> None:
    """Insert a single feature entry into the database if it does not already exist.

    Args:
        db_path (str): Path to the SQLite database.
        key (str): The unique identifier for this entry.
        feature (Dict[str, Any]): The extracted features to insert.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the entry already exists
    cursor.execute("SELECT 1 FROM features WHERE id = ?", (key,))
    if cursor.fetchone():
        logging.debug(f"Skipping existing key: {key}")
        conn.close()
        return

    cursor.execute("""
        INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        key,
        feature['complexity']['word_count'],
        feature['complexity']['unique_word_count'],
        feature['complexity']['comma_count'],
        feature['complexity']['complexity_score'],
        feature['modifiers']['num_adjectives'],
        feature['modifiers']['num_named_entities'],
        json.dumps(feature['modifiers']['adjectives']),
        json.dumps(feature['modifiers']['named_entities']),
        feature['categorical_features']['art_style'],
        feature['categorical_features']['camera_angle'],
        feature['categorical_features']['lighting'],
        feature['categorical_features']['time_of_day'],
        feature['categorical_features']['color'],
        feature['categorical_features']['color_palette'],
        feature['categorical_features']['emotion_mood'],
        feature['categorical_features']['explicitness'],
        feature['categorical_features']['composition_modifiers'],
        feature['categorical_features']['known_template']
    ))

    conn.commit()
    conn.close()
    logging.debug(f"Inserted features for key: {key}")



def extract_features_from_json(file_paths: list[str], db_path: str) -> None:
    """Extract features from multiple JSON datasets and store them in a SQLite database.

    Args:
        file_paths (list[str]): List of paths to the input JSON files.
        db_path (str): Path to the output SQLite database file.
    """
    combined_data = {}

    # Load and combine data from all files
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            combined_data.update(data)

    # Take only top 3 data points for testing
    combined_data = {k: combined_data[k] for k in list(combined_data)}

    create_database(db_path)

    for key, value in tqdm(combined_data.items(), desc="Extracting and saving features"):
        prompt = value.get('p', '')
        complexity = calculate_complexity(prompt)
        modifiers = count_modifiers_and_entities(prompt)
        categorical_features = extract_categorical_features(prompt)

        feature = {
            'complexity': complexity,
            'modifiers': modifiers,
            'categorical_features': categorical_features
        }

        insert_feature(db_path, key, feature)


if __name__ == "__main__":
    file_paths = [
        './data/images/part-000023/part-000023.json',
        './data/images/part-000024/part-000024.json',
        './data/images/part-000025/part-000025.json',
    ]
    db_path = './extracted_features.db'
    
    extract_features_from_json(file_paths, db_path)
