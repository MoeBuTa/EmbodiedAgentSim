#!/usr/bin/env python3
"""
Convert HM3D EQA CSV to JSON format expected by Habitat-Lab
"""

import pandas as pd
import json
import ast
from pathlib import Path

def convert_csv_to_json():
    # Read CSV
    csv_path = Path("/Users/wenxiao/PycharmProjects/EmbodiedAgentSim/habitat-lab/data/datasets/eqa/hm3d/hm3d-eqa/hm3d-eqa.csv")
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} rows in CSV")
    print("Columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head(3))
    
    # Convert to JSON format expected by Habitat, preserving original CSV attributes
    episodes = []
    for idx, row in df.iterrows():
        # Parse choices (assuming it's a string representation of a list)
        try:
            choices = ast.literal_eval(row['choices'])
        except:
            # If parsing fails, split by comma or use raw string
            choices = [row['choices']]
            
        # Get answer index (A=0, B=1, C=2, D=3)
        answer_idx = ord(row['answer']) - ord('A') if isinstance(row['answer'], str) and len(row['answer']) == 1 else 0
        
        episode = {
            'episode_id': str(idx),
            'scene_id': f"hm3d/train/{row['scene']}",
            'question': {
                'question_text': row['question'],
                'answer_text': choices[answer_idx] if answer_idx < len(choices) else choices[0],
                'question_tokens': row['question_formatted'].split(),
                'answer_token': answer_idx,
                'question_type': row.get('label', 'unknown')
            },
            'start_position': [0.0, 0.0, 0.0],  # Default positions - update with actual data if available
            'start_rotation': [1.0, 0.0, 0.0, 0.0],
            # Preserve original CSV attributes
            'scene': row['scene'],
            'floor': int(row['floor']) if pd.notna(row['floor']) else None,
            'choices': choices,
            'question_formatted': row['question_formatted'],
            'label': row['label']
        }
        episodes.append(episode)
        
        if idx < 3:  # Print first few for debugging
            print(f"\nEpisode {idx}:")
            print(f"  Scene: {episode['scene_id']}")
            print(f"  Question: {episode['question']['question_text']}")
            print(f"  Answer: {episode['question']['answer_text']}")
    
    # Save as JSON
    json_path = csv_path.parent / "hm3d-eqa.json"
    dataset = {
        'episodes': episodes
    }
    
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nCreated {json_path} with {len(episodes)} episodes")
    return json_path

if __name__ == "__main__":
    convert_csv_to_json()