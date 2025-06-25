

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

### Logs
```
INFO:__main__:Creating custom-llama3-1b model...
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
:
:
{'loss': 9.9372, 'grad_norm': 1.987466812133789, 'learning_rate': 3.96e-06, 'epoch': 0.26}
{'loss': 8.0194, 'grad_norm': 1.202463984489441, 'learning_rate': 7.960000000000002e-06, 'epoch': 0.52}
{'loss': 6.9469, 'grad_norm': 0.8132050633430481, 'learning_rate': 1.196e-05, 'epoch': 0.79}
{'loss': 5.4826, 'grad_norm': 0.7261037826538086, 'learning_rate': 1.5960000000000003e-05, 'epoch': 1.05}
{'loss': 3.9601, 'grad_norm': 1.1020852327346802, 'learning_rate': 1.9960000000000002e-05, 'epoch': 1.31}
{'eval_loss': 3.254647970199585, 'eval_runtime': 36.9631, 'eval_samples_per_second': 18.965, 'eval_steps_per_second': 4.762, 'epoch': 1.31}                     
 44%|█████████████████▍                      | 500/1146 [21:04<26:20,  2.45s/itINFO:__main__:█████████████████████████████████| 176/176 [00:36<00:00,  4.53it/s]
🔍 Prompt Evaluation @ step 500
INFO:__main__:📝 Prompt: Once upon a time
INFO:__main__:🧠 Model Output: Once upon a time time time time " time " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "
INFO:__main__:--------------------------------------------------
INFO:__main__:📝 Prompt: The capital of France is
INFO:__main__:🧠 Model Output: The capital of France is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is
INFO:__main__:--------------------------------------------------
{'loss': 2.6761, 'grad_norm': 0.7969480156898499, 'learning_rate': 1.693498452012384e-05, 'epoch': 1.57}                                                        
{'loss': 1.9135, 'grad_norm': 0.7001822590827942, 'learning_rate': 1.3839009287925698e-05, 'epoch': 1.83}
 67%|██████████████████████████▋             | 764/1146 [33:47<14:24,  2.26s/it]/Users/gitesh.grover/Study/AI-ERA/IntervueAI_Capstone/.venv/lib/python3.13/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 1.4545, 'grad_norm': 0.6565252542495728, 'learning_rate': 1.0743034055727555e-05, 'epoch': 2.09}
{'loss': 1.1952, 'grad_norm': 0.5984150171279907, 'learning_rate': 7.647058823529411e-06, 'epoch': 2.36}
{'loss': 1.0393, 'grad_norm': 0.526271641254425, 'learning_rate': 4.551083591331269e-06, 'epoch': 2.62}
{'eval_loss': 1.0282193422317505, 'eval_runtime': 38.9111, 'eval_samples_per_second': 18.015, 'eval_steps_per_second': 4.523, 'epoch': 2.62}                    
 87%|██████████████████████████████████     | 1000/1146 [44:34<06:12,  2.55s/itINFO:__main__:█████████████████████████████████| 176/176 [00:38<00:00,  4.53it/s]
🔍 Prompt Evaluation @ step 1000
INFO:__main__:📝 Prompt: Once upon a time
INFO:__main__:🧠 Model Output: Once upon a time time time time time time time time time when time when time when time when time when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when
INFO:__main__:--------------------------------------------------
INFO:__main__:📝 Prompt: The capital of France is
INFO:__main__:🧠 Model Output: The capital of France is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is
INFO:__main__:--------------------------------------------------
{'loss': 0.9675, 'grad_norm': 0.5134572386741638, 'learning_rate': 1.4551083591331269e-06, 'epoch': 2.88}                                                       
{'train_runtime': 3050.3713, 'train_samples_per_second': 6.003, 'train_steps_per_second': 0.376, 'train_loss': 3.8416107037928717, 'epoch': 3.0}
100%|███████████████████████████████████████| 1146/1146 [50:50<00:00,  2.66s/it]
INFO:__main__:Saving final model...
INFO:__main__:!!! FINISHED !!!
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:               eval/loss █▁
wandb:            eval/runtime ▁█
wandb: eval/samples_per_second █▁
wandb:   eval/steps_per_second █▁
wandb:             train/epoch ▁▂▂▃▄▄▄▅▆▆▇▇██
wandb:       train/global_step ▁▂▂▃▄▄▄▅▆▆▇▇██
wandb:         train/grad_norm █▄▂▂▄▂▂▂▁▁▁
wandb:     train/learning_rate ▂▃▅▆█▇▆▅▃▂▁
wandb:              train/loss █▇▆▅▃▂▂▁▁▁▁
wandb: 
wandb: Run summary:
wandb:                eval/loss 1.02822
wandb:             eval/runtime 38.9111
wandb:  eval/samples_per_second 18.015
wandb:    eval/steps_per_second 4.523
wandb:               total_flos 9809482418749440.0
wandb:              train/epoch 3
wandb:        train/global_step 1146
wandb:          train/grad_norm 0.51346
wandb:      train/learning_rate 0.0
wandb:               train/loss 0.9675
wandb:               train_loss 3.84161
wandb:            train_runtime 3050.3713
wandb: train_samples_per_second 6.003
wandb:   train_steps_per_second 0.376
wandb: 
wandb: 🚀 View run custom-llama3-1b-pretraining at: https://wandb.ai/ggrover-farzi-na/llama3-training/runs/4z2erlbt
wandb: ⭐️ View project at: https://wandb.ai/ggrover-farzi-na/llama3-training
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250624_001401-4z2erlbt/logs
```