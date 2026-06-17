import os
import json
import torch
from tqdm import tqdm
from transformers import pipeline

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIR = "./Indian Food Images/Indian Food Images"
OUTPUT_FILE = "llm_neighborhoods.json"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def get_classes(data_dir):
    """Extracts sorted class names from the dataset directory."""
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def generate_neighborhoods(classes, pipe):
    neighborhoods = {}
    
    for cls in tqdm(classes, desc="Generating LLM Neighborhoods"):
        clean_name = cls.replace("_", " ")
        
        # System prompt explicitly asking for JSON with comma-separated strings
        prompt = f"""You are an expert culinary AI. 
For the Indian food '{clean_name}', provide 5 broad semantic super-categories (Coarse) and 10 visually similar Indian dishes (Fine).
You MUST output ONLY valid JSON. The JSON must have exactly two keys: "coarse" and "fine".
The values for these keys must be a single comma-separated string containing the items.

Example format:
{{
  "coarse": "Indian food, dish, meal, cuisine, staple",
  "fine": "curry, bread, sweet, snack, rice, dal, biryani, dosa, roti, paneer"
}}

Output your JSON for '{clean_name}' below:
"""
        try:
            # Generate the response
            out = pipe(prompt, max_new_tokens=500, return_full_text=False)[0]['generated_text']
            print(cls)
            print(out)
            # Clean up potential markdown formatting the LLM might add
            json_str = out.strip()
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            elif json_str.startswith("```"):
                json_str = json_str.replace("```", "").strip()
                
            data = json.loads(json_str)
            
            c_list = [c.strip() for c in data.get("coarse", "").split(",") if c.strip()]
            f_list = [f.strip() for f in data.get("fine", "").split(",") if f.strip()]
            
            if not c_list: c_list = ["Indian food", "dish", "meal", "cuisine", "food"]
            if not f_list: f_list = ["curry", "bread", "sweet", "snack", "rice"]
            
            neighborhoods[cls] = {"C": c_list, "F": f_list}
            
        except Exception as e:
            raise e
            print(f"\n[Warning] Failed to parse JSON for {cls}: {e}. Using safe fallback.")
            neighborhoods[cls] = {
                "C": ["Indian food", "dish", "meal", "cuisine", "food"],
                "F": ["curry", "bread", "sweet", "snack", "rice"]
            }
            raise "skhdbfhjs"
            
    return neighborhoods

if __name__ == "__main__":
    print(f"Loading {MODEL_NAME} (this might take a while)...")
    pipe = pipeline(
        "text-generation", 
        model=MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    classes = get_classes(DATASET_DIR)
    print(f"Found {len(classes)} classes. Starting generation...")
    
    results = generate_neighborhoods(classes, pipe)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    