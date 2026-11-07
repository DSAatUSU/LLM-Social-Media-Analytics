import json
import pandas as pd

# Load JSON data
with open('generated_tweets_qualtrics.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare data for Excel
excel_data = []
for user in data:
    twitter_handle = user['twitter_handle']
    user_id = user['user_id']
    generated_tweets = user.get('generated_tweets', {})
    
    for llm_model, tweets in generated_tweets.items():
        for tweet in tweets:
            excel_data.append({
                'User ID': user_id,
                'Twitter Handle': twitter_handle,
                'LLM Model': llm_model,
                'Generated Tweet': tweet
            })

# Create DataFrame
df = pd.DataFrame(excel_data)

# Save to Excel (without encoding parameter)
output_excel = 'generated_tweets_output_21.xlsx'
df.to_excel(output_excel, index=False, engine='openpyxl')

print(f"Data successfully saved to {output_excel}")
