import requests
import json

def generate_response(prompt, model, temperature=0.7, max_tokens=1000):
    """
    Generate a response using the Ollama API with smolLM2:1.7b model.
    
    Args:
        prompt (str): The input prompt for the model
        model (str): The model name to use
        temperature (float): Controls randomness in the output (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The generated response
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Ollama API: {e}")
        return None

def main():
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence."
    ]
    
    print("Testing smolLM2:1.7b model with Ollama...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}:")
        print(f"Prompt: {prompt}")
        print("\nGenerating response...\n")
        
        # model = "llama3.2:latest"
        model= "smolLM2:1.7b"
        response = generate_response(prompt=prompt, model=model)
        
        if response:
            print(f"Response: {response}\n")
            print("-" * 80 + "\n")
        else:
            print("Failed to generate response.\n")
            print("-" * 80 + "\n")

if __name__ == "__main__":
    main() 