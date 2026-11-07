import openai
import google.generativeai as genai
import ollama
import requests
from openai import OpenAI

def initialize_model(model_type, model_name=None, api_key=None, gemini_api_key=None, deepseek_api_key=None):
    """
    Initialize the specified model type for content generation.
    """
    if model_type == "openai":
        if api_key is None:
            raise ValueError("API key is required for OpenAI.")
        return {"api_key": api_key}  # ✅ Store API key properly

    elif model_type == "gemini":
        if gemini_api_key is None:
            raise ValueError("API key is required for Gemini.")
        genai.configure(api_key=gemini_api_key)
        return None

    elif model_type == "ollama":
        if model_name is None:
            raise ValueError("Model name is required for Ollama.")
        ollama.pull(model_name)
        return {"model_name": model_name}

    elif model_type == "deepseek":
        if deepseek_api_key is None:
            raise ValueError("API key is required for DeepSeek.")
        return {"api_key": deepseek_api_key, "api_base": "https://api.deepseek.com"}

    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose 'openai', 'gemini', 'ollama', or 'deepseek'.")

def generate_tweets(input_data, model_name, model_type, num_generated_tweets=10, temperature=0.7, model_resources=None, sampling_number=50):
    generated_tweets = []

    sampled_tweets = input_data["sampled_tweets"]
    user_context = input_data["user_context"]

    prompt = (
        f"We have collected {sampling_number} tweets from one user on Twitter, included below:\n\n"
        f"{chr(10).join(sampled_tweets)}\n\n"
        "Below, you can also find the user-written bio (description) as well as the number of followings and followers of the user:\n\n"
        f"User Bio: {user_context['description']}\n"
        f"Number of Followings: {user_context['following_count']}\n"
        f"Number of Followers: {user_context['followers_count']}\n\n"
        f"Now, generate exactly {num_generated_tweets} tweets that the user could have posted on their Twitter account. "
        "Each tweet should not exceed 280 characters and must be formatted as:\n"
        "1. <tweet_text>\n"
        "2. <tweet_text>\n"
        "...\n"
        f"Ensure all {num_generated_tweets} tweets are included, without extra spacing or missing tweets."
    )

    if model_type == "openai":
        try:
            client = OpenAI(api_key=model_resources["api_key"])

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": "You are a tweet generator."},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=280 * num_generated_tweets  # ✅ Ensure enough tokens
            )

            output_text = response.choices[0].message.content.strip()

            # ✅ Filter out empty lines & ensure the right number of tweets
            tweets = [line.strip() for line in output_text.split("\n") if line.strip()]
            generated_tweets = tweets[:num_generated_tweets]

            # ✅ Retry if fewer tweets are generated
            if len(generated_tweets) < num_generated_tweets:
                print(f"⚠️ Warning: Only {len(generated_tweets)} tweets generated. Retrying...")
                return generate_tweets(input_data, model_name, model_type, num_generated_tweets, temperature, model_resources, sampling_number)

        except openai.OpenAIError as e:
            print(f"OpenAI API Error: {e}")

    elif model_type == "gemini":
        try:
            model = genai.GenerativeModel("gemini-1.5-pro-002")
            response = model.generate_content(prompt, generation_config={"temperature": temperature})
            output_text = response.text.strip()
            tweets = [line.strip() for line in output_text.split("\n") if line.strip()]
            generated_tweets = tweets[:num_generated_tweets]

        except Exception as e:
            print(f"Gemini API Error: {e}")

    elif model_type == "ollama":
        if not model_resources:
            raise ValueError("Model resources are required for 'ollama'.")
        model_name = model_resources["model_name"]

        try:
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}], options={"temperature": temperature})

            if hasattr(response, "message") and hasattr(response.message, "content"):
                output_text = response.message.content.strip()
                tweets = [line.strip() for line in output_text.split("\n") if line.strip()]
                generated_tweets = tweets[:num_generated_tweets]

        except Exception as e:
            print(f"Error with Ollama content generation: {e}")

    elif model_type == "deepseek":
        try:
            client = OpenAI(api_key=model_resources["api_key"], base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=False
            )

            if hasattr(response, "choices") and len(response.choices) > 0:
                if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "content"):
                    output_text = response.choices[0].message.content.strip()
                    tweets = [line.strip() for line in output_text.split("\n") if line.strip()]
                    generated_tweets = tweets[:num_generated_tweets]

        except Exception as e:
            print(f"DeepSeek API Error: {e}")

    return generated_tweets
