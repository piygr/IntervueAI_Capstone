

## 🤖 Model Implementations

### Llama3 1B Model

The project includes a custom implementation of the Llama3 1B model in `pre_training/models/llama3.py`. This implementation features:

#### Architecture
- Hidden size: 1024
- Intermediate size: 2816
- Number of layers: 12
- Number of attention heads: 16
- Vocabulary size: 32000
- Maximum sequence length: 2048

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

The model can be instantiated using the `create_llama3_1b()` function, which returns a properly configured Llama3 model with approximately 1B parameters.

---

## 🚀 Training

### Overview
1. **Dataset Selection:**
- Using WikiText-2 dataset, which is a high-quality, curated dataset
- Proper train-validation split (90-10) to monitor model performance

2. **Overfitting Prevention Measures:**
- L2 Regularization (weight_decay=0.01)
- Early Stopping (patience=3, threshold=0.01)
- Gradient Clipping (max_grad_norm=1.0)
- Mixed Precision Training (fp16=True)
- Learning Rate Warmup (warmup_steps=500)
- Model Checkpointing (saves best model based on validation loss)

3. **Training Configuration:**
- Small batch size (4) with gradient accumulation (4 steps)
- Learning rate of 2e-5
- 3 epochs maximum
- Regular evaluation on validation set
- Weights & Biases integration for monitoring

4. **Model Architecture:**
- Using Llama-2-7b as the base model
- Added special tokens for better text processing
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
python pre_training/train_llama3.py

### Important notes:
- You'll need a GPU with at least 16GB VRAM to train this model
- The training will take several hours depending on your hardware
- The script will save checkpoints in the ./llama3_checkpoints directory
- The final model will be saved in ./llama3_final_model

The script includes several safeguards against overfitting:
- Regular validation checks
- Early stopping if the model stops improving
- L2 regularization to prevent large weights
- Gradient clipping to prevent exploding gradients
- Learning rate warmup for stable training