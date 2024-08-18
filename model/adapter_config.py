from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Function to attach a LoRA (Low-Rank Adaptation) adapter to the model for efficient fine-tuning
def attach_adapter(model, rank=None, alpha=None, target_modeules=None):
    # If no specific rank is provided, use a default value
    if rank is None:
        lora_r = 2048  # Default rank for LoRA
    else:
        lora_r = rank  # Use the provided rank value

    # If no specific alpha is provided, use the rank value as the default
    if alpha is None:
        lora_alpha = rank  # Use rank as default alpha if not provided
    else:
        lora_alpha = alpha  # Use the provided alpha value

    # Define the target modules to which LoRA adapters will be applied
    if target_modeules is None:
        target_modules = [
            "o_proj",  # Output projection
            "v_proj",  # Value projection
            "q_proj",  # Query projection
            "k_proj",  # Key projection
        ]
    else:
        target_modules = target_modeules  # Use the provided target modules

    lora_dropout = 0.2  # Set a dropout rate to prevent overfitting

    modules_to_save = ["lm_head", "embed_tokens"]  # Define the modules that should be saved during training

    # Configure the LoRA adapter with the specified parameters
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,  # Scale factor for the low-rank adaptation
        lora_dropout=lora_dropout,  # Dropout rate for LoRA
        r=lora_r,  # Rank for the low-rank matrices
        target_modules=target_modules,  # List of modules where LoRA will be applied
        bias="none",  # No bias adjustment for LoRA
        modules_to_save=modules_to_save,  # Modules to save during training
        task_type="CAUSAL_LM"  # Task type is causal language modeling
    )
    
    # Prepare the model for training with low-bit precision (e.g., 8-bit, 4-bit)
    model = prepare_model_for_kbit_training(model)
    
    # Attach the LoRA adapter to the model
    model = get_peft_model(model, peft_config)
    
    # Print the number of trainable parameters in the model
    model.print_trainable_parameters()
    
    return model
