import pandas as pd
import json
from pathlib import Path

# Define input feather file and output JSON file
feather_file = Path("./prompts_keywords.feather")  # Update this path
json_file = Path("cache_output.json")

# Load the feather file into a DataFrame
df = pd.read_feather(feather_file)

# Convert the DataFrame into a dictionary
# If `keywords` column contains lists/objects as strings, you might need to `eval()` them
prompt_to_keywords = dict(zip(df["prompt"], df["keywords"]))

# Save to a JSON file
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(prompt_to_keywords, f, indent=4, ensure_ascii=False)

print(f"Converted Feather to JSON and saved to: {json_file}")
