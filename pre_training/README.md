

## 🤖 Model Implementations

### Custom Llama3 Model

The project includes a custom implementation of the Llama3 1B model in `pre_training/models/llama3.py`. This implementation use the configurable config.


#### Key Features
- SwiGLU activation function
- Parallel attention mechanism
- Grouped query attention
- Rotary position embeddings
- KV cache for efficient inference
- RMSNorm for layer normalization

#### Model Components
1. **Attention Module**
   - Multi-headed attention with grouped query attention
   - Rotary position embeddings
   - KV cache support for efficient inference

2. **MLP Module**
   - SwiGLU activation
   - Gate and up projections
   - Efficient down projection

3. **Decoder Layer**
   - Self-attention with residual connection
   - MLP with residual connection
   - Layer normalization

4. **Model Architecture**
   - Token embeddings
   - Stack of decoder layers
   - Final layer normalization
   - Language modeling head


## Model Configurations
The model can be instantiated using the `create_llama3_1b()` function, which returns a properly configured Llama3 model. It takes a mandatory argument `config_type` that selects one of the predefined configuration for the modeland dictates the total number of parameters of the model. Currently, there are 3 configs to choose from:
<ul>
<li><b>0.5B</b> - approx 0.4-0.5 B params</li>
<li><b>1B</b> - approx 1B params</li>
<li><b>1.5B</b> - approx 1.5-1.6 B params</li>
</ul>
In addition, it takes an optional `input_config` that overrides the model's config. It is mainly useful to supply tokenizer related properties such as `vocab_size`, `pad_token_id`, etc

Also, there is another `loadLlamaModelWithoutWeights()` method  that instantiates the one of the existing llama3 model from the hugging space without the trained weights. The model type is dicatated by the `model_type` argument. In addition, it takes the same `config_type` and `input_config` arguments in addition (similar to the other method). 

---

## 🚀 Training

### Overview
The main `train_llama3.py` takes bunch of optional arguments that decides which model to invoke with what config:
* model_type:
    * custom-llama3 (default):  Loads the custom model for training
    * meta-llama/Llama-3.2-1B: Loads the metallama model.
* config_type:
    * 0.5B (default): Loads the model with the config that totals the parameters around 400-500 million
    * 1B: Loads the model with the confg that totals the parameters around 1B
    * 1.5B: Loads the model with the config that totals the parameters around 1.5B
* use_llama2_tokenizer:
    * False (default): Whether to use default llama3 tokenizer of vocab_size 128256
    * True: Whether to use llana2 type tokenizer with 32k vocab size 
Example:
```bash
# This will train the custom llama3 model with 0.5B config using llama3 tokenizer
python pre_training/train_llama3.py
```

```bash
# This will train the meta llama3 model with 1B config using llama3 tokenizer
python pre_training/train_llama3.py --model_type meta-llama/Llama-3.2-1B --config_type 1B --use_llama2_tokenizer False
```
### **Dataset Selection:**
- Using WikiText-2 dataset, which is a high-quality, curated dataset
- Proper train-validation split to monitor model performance

2. **Overfitting Prevention Measures:**
- L2 Regularization (weight_decay=0.01)
- Gradient Clipping (max_grad_norm=1.0)
- Mixed Precision Training (fp16=True)
- Learning Rate Warmup (warmup_steps=500)
- Model Checkpointing (saves best model based on validation loss)

3. **Training Configuration:**
- Small batch size (4) with gradient accumulation (4 steps)
- Learning rate of 2e-5
- 3 epochs maximum
- Regular evaluation on validation set and manual prompts
- Weights & Biases integration for monitoring

4. **Model Architecture:**
- Using Llama-3-1b as the base model
- Adds special tokens for better text processing if needed
- Proper tokenizer configuration

### Install depedencies
Ensure you are in `pre_training` directory
```bash
cd IntervueAI_Capstone/pre_training
pip install -r requirements.txt
```
### Set up Weights & Biases (optional but recommended):
```bash
wandb login
```

### Run the training script:
```bash
# This will train the custom llama3 model with 0.5B config using llama3 tokenizer
python pre_training/train_llama3.py
```
or
```bash
# This will train the meta llama3 model with 1B config using llama3 tokenizer
python pre_training/train_llama3.py --model_type meta-llama/Llama-3.2-1B --config_type 1B --use_llama2_tokenizer False
```

### Important notes:
- The training will take several hours depending on your hardware
- The script will save checkpoints in the ./llama3_checkpoints directory
- The final model will be saved in ./llama3_final_model

The script includes several safeguards against overfitting:
- Regular validation checks
- Early stopping if the model stops improving
- L2 regularization to prevent large weights
- Gradient clipping to prevent exploding gradients
- Learning rate warmup for stable training

### Logs
We ran multiple rounds of trainings. When we tried to run it with 1B config or more, we have exhausing local system's memory. So, we eneded up training with only 0.5B config for quite a few hours to check that the loss is decreasing as expected. We tried to run it for both custom and meta's llm model to check they have been performing similar around the similar steps.
These logs can be found in the `training_logs` sub-directory.
