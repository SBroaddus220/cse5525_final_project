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

    features = {
        "art_style": next((style for style in ["romanesque", "baroque", "anime", "pixel", "surreal", "realistic", "steampunk"] if style in prompt), "unspecified"),
        "camera_angle": next((angle for angle in ["close-up", "top-down", "side view", "overhead"] if angle in prompt), "unspecified"),
        "lighting": next((light for light in ["studio lighting", "dramatic lighting", "natural light", "backlit", "soft lighting", "volumetric", "sharp focus"] if light in prompt), "unspecified"),
        "time_of_day": next((time for time in ["sunset", "dawn", "night", "daytime", "noon", "morning", "day", "afternoon", "nighttime"] if time in prompt), "unspecified"),
        "color": next((color for color in ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white"] if color in prompt), "unspecified"),
        "color_palette": next((palette for palette in ["vibrant", "muted", "monochrome", "pastel", "colorful", "sepia", "high contrast", "coherent"] if palette in prompt), "unspecified"),
        "emotion_mood": next((mood for mood in ["peaceful", "dramatic", "eerie", "joyful", "melancholic", "awe"] if mood in prompt), "neutral"),
        "explicitness": "explicit" if any(word in prompt for word in ["highly detailed", "realistic", "intricate", "ultra detailed", "hyper detailed", "sharp focus", "octane render", "volumetric", "ray tracing"]) else "implicit",
        "composition_modifiers": "yes" if any(word in prompt for word in ["centered", "rule of thirds", "symmetrical", "zoomed in", "wide shot", "sharp focus"]) else "no",
        "known_template": "artstation_trending" if "artstation trending" in prompt else "freeform"
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



def extract_features_from_json(file_path: str, db_path: str) -> None:
    """Extract features from a JSON dataset and store them in a SQLite database.

    Args:
        file_path (str): Path to the input JSON file.
        db_path (str): Path to the output SQLite database file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Take only top 3 data points for testing
    # data = {k: data[k] for k in list(data)[:3]}

    create_database(db_path)

    for key, value in tqdm(data.items(), desc="Extracting and saving features"):
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
    file_path = './data/images/part-000023/part-000023.json'
    db_path = './extracted_features.db'
    
    extract_features_from_json(file_path, db_path)
