import json
import re
import spacy

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
    

def extract_features_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    features = {}
    for key, value in data.items():
        prompt = value.get('p', '')
        complexity = calculate_complexity(prompt)
        modifiers = count_modifiers_and_entities(prompt)
        categorical_features = extract_categorical_features(prompt)
        features[key] = {
            'complexity': complexity,
            'modifiers': modifiers,
            'categorical_features': categorical_features
        }

    return features


if __name__ == "__main__":
    file_path = './data/images/part-000023/part-000023.json'
    features = extract_features_from_json(file_path)
    if features:
        print("Extracted Features:", features)
        