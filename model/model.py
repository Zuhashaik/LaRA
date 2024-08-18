from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Function to load a language model with optional quantization for reduced memory usage
def load_model(model_name=None, quantize=True):
    # If no specific model name is given, use a default model
    if model_name:
        model_name_or_path = model_name
    else:
        model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'  # Default model

    # Load the model with 4-bit quantization to save memory
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Compute type to reduce precision and save memory
        )
        # Load the model with quantization enabled
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',  # Automatically decide where to load the model (CPU/GPU)
            quantization_config=bnb_config, 
        )
    else:
        # Load the model without quantization (uses more memory)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto'
        )
        
    # Load the tokenizer, which is used to process text data for the model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    # Add a special padding token to the tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return model, tokenizer

# Function to resize the word embedding layer to include new tokens
def resize_WEL(model, tokenizer, special_tokens=None, additional_token=None):
    # Check the current size of the model's word embedding matrix (how many words it can represent)
    before_len = model.model.embed_tokens.weight.shape
    
    # Combine special and additional tokens if provided, otherwise use default tokens
    if (special_tokens != None and additional_token != None):
        new_tokens = special_tokens + additional_token
    else:
        # Create a list of new tokens, such as <0>, <1>, ... <999>, <txt>, etc.
        new_tokens = [f"<{x}>" for x in range(1000)] + ['<txt>', '</txt>', '<sp>', '</sp>']
        tokenizer.add_tokens(new_tokens)  # Add these new tokens to the tokenizer
    
    # Resize the model's word embedding layer to accommodate the new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Check the size of the embedding matrix after resizing
    after_len = model.model.embed_tokens.weight.shape
    
    # Print out a message showing the change in size
    print(f'Model word embedding matrix has resized from {before_len} to {after_len}')
    
    return model, tokenizer
