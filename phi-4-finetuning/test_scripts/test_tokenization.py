import os
from src.data_preparation.tokenize import Tokenizer

try:
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Tokenizer()
    
    # Test tokenizing a sample message
    print("Testing tokenization with a sample message...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
    ]
    
    # Convert messages to prompt
    prompt = tokenizer.convert_messages_to_prompt(messages)
    print(f"Converted prompt:\n{prompt}\n")
    
    # Tokenize the prompt
    tokens = tokenizer.tokenizer.encode(prompt)
    print(f"Token count: {len(tokens)}")
    print(f"First 10 tokens: {tokens[:10]}")
    
    # Test tokenizing a sample example
    example = {"messages": messages}
    tokenized = tokenizer.tokenize_example(example)
    
    print("✅ Tokenization test successful!")
    print(f"Input IDs length: {len(tokenized['input_ids'])}")
    print(f"Attention mask length: {len(tokenized['attention_mask'])}")
    
except Exception as e:
    print(f"❌ Tokenization test failed: {str(e)}")