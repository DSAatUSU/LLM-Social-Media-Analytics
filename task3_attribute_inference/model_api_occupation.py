import openai
import google.generativeai as genai
import ollama
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import json

def initialize_model(model_type, api_key=None, gemini_api_key=None, deepseek_api_key=None, model_name=None, fine_tuned_path=None, categories=None):
    if model_type == "openai":
        if api_key is None:
            raise ValueError("API key required for OpenAI.")
        return {"api_key": api_key}

    elif model_type == "gemini":
        if gemini_api_key is None:
            raise ValueError("API key required for Gemini.")
        genai.configure(api_key=gemini_api_key)
        return None

    elif model_type == "ollama":
        if model_name is None:
            raise ValueError("Model name required for Ollama.")
        ollama.pull(model_name)
        return {"model_name": model_name}

    elif model_type == "deepseek":
        if deepseek_api_key is None:
            raise ValueError("API key required for DeepSeek.")
        return {"api_key": deepseek_api_key, "api_base": "https://api.deepseek.com"}

    elif model_type == "bert":
        if fine_tuned_path:
            tokenizer = BertTokenizer.from_pretrained(fine_tuned_path)
            model = BertForSequenceClassification.from_pretrained(fine_tuned_path)
        else:
            if model_name is None or categories is None:
                raise ValueError("For default BERT, provide both 'model_name' and 'categories'.")
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(categories))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        categories = model.config.id2label

        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "categories": {int(i): label for i, label in categories.items()}
        }

    else:
        raise ValueError("Invalid model_type. Choose from 'openai', 'gemini', 'ollama', 'deepseek', or 'bert'.")

def classify_occupation(prompt, model_name, model_type, model_resources, temperature=0):
    if model_type == "openai":
        client = openai.OpenAI(api_key=model_resources["api_key"])
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Classify occupations into given categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    elif model_type == "gemini":
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        return response.text.strip()

    elif model_type == "ollama":
        response = ollama.chat(
            model=model_resources["model_name"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return response.message.content.strip()

    elif model_type == "deepseek":
        client = openai.OpenAI(api_key=model_resources["api_key"], base_url=model_resources["api_base"])
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Classify occupations into given categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    elif model_type == "bert":
        model = model_resources["model"]
        tokenizer = model_resources["tokenizer"]
        device = model_resources["device"]
        categories = model_resources["categories"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_label_id = logits.argmax(dim=1).item()

        return json.dumps({"occupation_category": categories[predicted_label_id]})

    else:
        raise ValueError("Invalid model_type. Choose from 'openai', 'gemini', 'ollama', 'deepseek', or 'bert'.")
