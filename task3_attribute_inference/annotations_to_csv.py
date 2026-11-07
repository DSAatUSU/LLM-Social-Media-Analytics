import json
import pandas as pd

# Load JSON data
with open('selected_users2_fully_labeled.json', 'r', encoding='utf-8') as f:
    users = json.load(f)

# Prepare CSV rows by concatenating tweets and descriptions
rows = []
for user in users:
    tweets_concat = ' '.join(user['tweets'])
    description = user.get('description', '')  # Safely get descriptions
    combined_text = tweets_concat + " " + description  # Combine tweets and description clearly

    interest_category = user['interests_hobbies']
    rows.append({
        "tweets": combined_text,
        "label": interest_category
    })

# Create DataFrame and save CSV
df = pd.DataFrame(rows)
df.to_csv('tweets_dataset.csv', index=False)

print("✅ CSV dataset created successfully with tweets and descriptions!")
