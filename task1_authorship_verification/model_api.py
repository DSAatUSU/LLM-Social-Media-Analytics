import openai
from transformers import BertForSequenceClassification, BertTokenizer
import google.generativeai as genai
import ollama
import torch
import re
import time
import requests
from openai import OpenAI

def extract_first_number(text):
    """ Extracts the first valid number (0 or 1) from a string. """
    match = re.search(r'\b[01]\b', text)  # Looks for isolated 0 or 1
    if match:
        return match.group()
    return None  # Return None if no valid number is found

def initialize_model(model_type, model_name=None, api_key=None, gemini_api_key=None, deepseek_api_key=None):
    """
    Initialize the specified model type.
    Args:
        model_type (str): Type of the model ("openai", "bert", "gemini", "ollama", "deepseek").
        model_name (str, optional): Name of the model to initialize (required for "ollama").
        api_key (str, optional): API key for OpenAI (required for "openai").
        gemini_api_key (str, optional): API key for Gemini (required for "gemini").
        deepseek_api_key (str, optional): API key for DeepSeek (required for "deepseek").
    Returns:
        dict or None: For local models (e.g., BERT, Llama), returns a dict containing the model and tokenizer.
                      For hosted APIs (e.g., OpenAI, Gemini, DeepSeek), returns None.
    """
    if model_type == "openai":
        if api_key is None:
            raise ValueError("API key is required for OpenAI.")
        openai.api_key = api_key
        return None
    elif model_type == "bert":
        model_name = "bert-base-uncased"
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        return {"model": model, "tokenizer": tokenizer}
    elif model_type == "gemini":
        if gemini_api_key is None:
            raise ValueError("API key is required for Gemini.")
        genai.configure(api_key=gemini_api_key)
        return None
    elif model_type == "ollama":
        if model_name is None:
            raise ValueError("Model name is required for Ollama.")
        ollama.pull(model_name)  # Ensure the model is pulled and ready
        return {"model_name": model_name}
    elif model_type == "deepseek":
        if deepseek_api_key is None:
            raise ValueError("API key is required for DeepSeek.")
        return {
            "api_key": deepseek_api_key,
            "api_base": "https://api.deepseek.com"
        }
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose 'openai', 'bert', 'gemini', 'ollama', or 'deepseek'.")

def classify_tweets(prompt, tweet, model_name, model_type, model_resources=None):
    """
    Classify a single tweet using OpenAI, BERT, Gemini, Ollama, or DeepSeek.
    Ensures only valid responses (0 or 1) are returned.
    """
    if model_type == "openai":
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a tweet classification assistant. Respond with only '0' or '1'."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            prediction = extract_first_number(response['choices'][0]['message']['content'])
            return {"tweet": tweet, "prediction": prediction if prediction else "0"}  # Default to 0 if invalid
        except openai.error.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return None

    elif model_type == "bert":
        if not model_resources:
            raise ValueError("Model resources are required for 'bert'.")
        model = model_resources["model"]
        tokenizer = model_resources["tokenizer"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        try:
            inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            prediction = outputs.logits.argmax(dim=1).item()
            return {"tweet": tweet, "prediction": str(prediction)}
        except Exception as e:
            print(f"Error with BERT classification: {e}")
            return None

    elif model_type == "gemini":
        try:
            role_description = "You are a tweet classification assistant. Respond with only '0' or '1'."
            extended_prompt = f"{role_description}\n\n{prompt}"

            model = genai.GenerativeModel("gemini-1.5-pro-002")
            response = model.generate_content(extended_prompt)

            prediction = extract_first_number(response.text)
            return {"tweet": tweet, "prediction": prediction if prediction else "0"}  # Default to 0
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return None

    elif model_type == "ollama":
        if not model_resources:
            raise ValueError("Model resources are required for 'ollama'.")
        model_name = model_resources["model_name"]

        try:
            messages = [
                {"role": "system", "content": "You are a tweet classification assistant. Respond with only '0' or '1'."},
                {"role": "user", "content": prompt}
            ]
            response = ollama.chat(model=model_name, messages=messages)

            if hasattr(response, "message") and hasattr(response.message, "content"):
                prediction = extract_first_number(response.message.content)
                return {"tweet": tweet, "prediction": prediction}
            else:
                print(f"Ollama response parsing error: {response}")
                return None
        except Exception as e:
            print(f"Error with Ollama classification: {e}")
            return None

    elif model_type == "deepseek":
        try:
            client = OpenAI(api_key=model_resources["api_key"], base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a tweet classification assistant. Respond with only '0' or '1'."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )

            # Validate the response structure before extracting prediction
            if hasattr(response, "choices") and len(response.choices) > 0:
                if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "content"):
                    prediction_text = response.choices[0].message.content.strip()
                    
                    # Extract first valid number (0 or 1)
                    prediction = extract_first_number(prediction_text)
                    return {"tweet": tweet, "prediction": prediction if prediction else "0"}  # Default to 0
                else:
                    print("DeepSeek API Error: Missing 'message' or 'content' in response.")
                    return None
            else:
                print("DeepSeek API Error: Unexpected response format:", response)
                return None
        except Exception as e:
            print(f"DeepSeek API Error: {e}")
            return None
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose 'openai', 'bert', 'gemini', 'ollama', or 'deepseek'.")
