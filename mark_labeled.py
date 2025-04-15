import os
import csv

# Dictionary mapping directory names to labels
label_mapping = {
    "bao ngu xam trang": 1,
    "dui ga baby": 2,
    "linh chi trang": 3,
    "nam mo": 0
}

# Path to the train directory (same level as the script)
train_dir = "train"

# Create csv directory if it doesn't exist
os.makedirs("csv", exist_ok=True)

# Path to output CSV file
output_csv = os.path.join("csv", "mushroom_labels.csv")

# List to store all data
data = []

# Iterate through each subdirectory in the train directory
for subdir in os.listdir(train_dir):
    # Get the full path of the subdirectory
    subdir_path = os.path.join(train_dir, subdir)
    
    # Skip if not a directory or not in our mapping
    if not os.path.isdir(subdir_path) or subdir not in label_mapping:
        continue
    
    # Get the label for this directory
    label = label_mapping[subdir]
    
    # Iterate through each file in the subdirectory
    for filename in os.listdir(subdir_path):
        # Extract just the filename without extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Add to our data list
        data.append([name_without_ext, label])

# Write data to CSV file
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(["id", "type"])
    # Write data
    writer.writerows(data)

print(f"Labels have been saved to {output_csv}")