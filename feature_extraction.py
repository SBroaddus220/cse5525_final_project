import json
import re
import spacy
from tqdm import tqdm

def extract_categorical_features(prompt):
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

    return features
    

def count_modifiers_and_entities(prompt):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(prompt)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    named_entities = [ent.text for ent in doc.ents]
    return {
        'num_adjectives': len(adjectives),
        'num_named_entities': len(named_entities),
        'adjectives': adjectives,
        'named_entities': named_entities,
    }


def calculate_complexity(prompt):
    words = re.findall(r'\b\w+\b', prompt)
    word_count = len(words)
    unique_word_count = len(set(words))
    comma_count = prompt.count(',')

    complexity_score = (
        0.3 * word_count +
        0.5 * unique_word_count +
        0.2 * comma_count
    )
    return {
        'word_count': word_count,
        'unique_word_count': unique_word_count,
        'comma_count': comma_count,
        'complexity_score': round(complexity_score,2)
    }
    


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
    combined_data = {k: combined_data[k] for k in list(combined_data)[:3]}

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
    db_path = './prompt_features.sqlite3'
    features = extract_features_from_json(file_paths, db_path)
    if features:
        print("Extracted Features:", features)
        # Dump to JSON file
        with open('extracted_features.json', 'w', encoding='utf-8') as outfile:
            json.dump(features, outfile, indent=4)
        