import os 
from openai import OpenAI
import time
import json
import re

from dotenv import load_dotenv
load_dotenv()

def load_log(LOG_FILE):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    else:
        return {"tokens_used": 0, "time_taken_last_run": 0.0, "time_taken_total": 0.0, "runs": 0}

def save_log(LOG_FILE, log_data):
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=4)

def query_deepseek_model(prompt, model, temperature, max_tokens, LOG_FILE):
    '''
    prompt = context + prompt
    model = 'deepseek-chat' for DeepSeek's chat model
    temperature: set the temperature for the model for determining the randomness of the output.
    max_tokens: set the maximum number of tokens to generate for the output.
    LOG_FILE: the log file to save the log data
    '''
    # Initialize OpenAI client with DeepSeek configuration
    client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
    except Exception as e:
        print(f'Encountered an error: {e}')
        return None, str(e)
    
    end_time = time.time()
    
    # Estimate tokens (since DeepSeek doesn't provide token count directly)
    # Rough estimate: 1 token ≈ 4 characters
    input_tokens = len(prompt) // 4
    response_content = response.choices[0].message.content
    output_tokens = len(response_content) // 4
    
    total_duration = end_time - start_time
    total_tokens = input_tokens + output_tokens
    
    if response_content:
        # save the log
        log_data = load_log(LOG_FILE)
        log_data["tokens_used"] += total_tokens
        log_data["time_taken_last_run"] = total_duration
        log_data["time_taken_total"] += total_duration
        log_data["runs"] += 1
        save_log(LOG_FILE, log_data)
        return response_content, None
    else:
        log_data = load_log(LOG_FILE)
        log_data["tokens_used"] += total_tokens
        log_data["time_taken_last_run"] = total_duration
        log_data["time_taken_total"] += total_duration
        log_data["runs"] += 1
        save_log(LOG_FILE, log_data)
        return None, "No response content" 