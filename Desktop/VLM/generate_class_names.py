import os
import json

data_dir = 'data'
class_names = []

for d in os.listdir(data_dir):
    path = os.path.join(data_dir, d)
    if os.path.isdir(path):
        # Only include if it has files (images)
        if len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]) > 0:
            class_names.append(d)

class_names.sort()

with open('models/class_names.json', 'w') as f:
    json.dump(class_names, f, indent=4)

print(f"Generated models/class_names.json with {len(class_names)} classes.")
