#!/usr/bin/env python3
import json

with open("/Users/wenxiao/PycharmProjects/EmbodiedAgentSim/habitat-lab/data/datasets/eqa/hm3d/express-bench/express-bench.json") as f:
    data = json.load(f)
    print(f"Express-bench episodes: {len(data)}")
    print("First episode structure:")
    first = data[0]
    for key in first.keys():
        if key != "actions":
            print(f"  {key}: {first[key]}")
        else:
            print(f"  actions: dict with {len(first[key])} steps")
    
    # Check question types
    types = set(ep.get("type", "unknown") for ep in data)
    print(f"Question types: {types}")