

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
wandb: ⭐️ View project at https://wandb.ai/ggrover-farzi-na/llama3-training
wandb: 🚀 View run at https://wandb.ai/ggrover-farzi-na/llama3-training/runs/k28k270x
:
:
{'loss': 1.8148, 'grad_norm': 0.4898901879787445, 'learning_rate': 3.96e-06, 'epoch': 0.06}
{'loss': 1.5842, 'grad_norm': 0.6487119793891907, 'learning_rate': 7.960000000000002e-06, 'epoch': 0.12}
{'loss': 1.477, 'grad_norm': 0.3755301535129547, 'learning_rate': 1.196e-05, 'epoch': 0.18}
{'loss': 1.3317, 'grad_norm': 0.5248742699623108, 'learning_rate': 1.5960000000000003e-05, 'epoch': 0.24}
{'loss': 1.0687, 'grad_norm': 0.6217405796051025, 'learning_rate': 1.9960000000000002e-05, 'epoch': 0.31}
{'loss': 0.9301, 'grad_norm': 0.6149572730064392, 'learning_rate': 1.95514272768464e-05, 'epoch': 0.37}
{'loss': 0.8182, 'grad_norm': 0.8558429479598999, 'learning_rate': 1.9098323516085183e-05, 'epoch': 0.43}
{'loss': 0.6519, 'grad_norm': 0.7973967790603638, 'learning_rate': 1.8645219755323973e-05, 'epoch': 0.49}                    
{'loss': 0.5867, 'grad_norm': 0.7576380372047424, 'learning_rate': 1.8192115994562755e-05, 'epoch': 0.55}                    
{'loss': 0.5037, 'grad_norm': 1.102279782295227, 'learning_rate': 1.773901223380154e-05, 'epoch': 0.61}                      
{'loss': 0.4393, 'grad_norm': 0.9205381274223328, 'learning_rate': 1.7285908473040327e-05, 'epoch': 0.67}                    
{'loss': 0.3969, 'grad_norm': 0.8330844640731812, 'learning_rate': 1.6832804712279113e-05, 'epoch': 0.73}                    
{'loss': 0.3588, 'grad_norm': 0.6686252355575562, 'learning_rate': 1.63797009515179e-05, 'epoch': 0.79}                      
{'loss': 0.299, 'grad_norm': 0.6245916485786438, 'learning_rate': 1.5926597190756684e-05, 'epoch': 0.85}                     
{'loss': 0.2835, 'grad_norm': 0.884012758731842, 'learning_rate': 1.547349342999547e-05, 'epoch': 0.92}                      
{'loss': 0.2504, 'grad_norm': 0.8276767730712891, 'learning_rate': 1.5020389669234256e-05, 'epoch': 0.98}                    
{'loss': 0.2245, 'grad_norm': 0.8295820355415344, 'learning_rate': 1.456728590847304e-05, 'epoch': 1.04}                     
{'loss': 0.1998, 'grad_norm': 0.818139910697937, 'learning_rate': 1.4114182147711828e-05, 'epoch': 1.1}                      
{'loss': 0.1875, 'grad_norm': 0.4903353452682495, 'learning_rate': 1.3661078386950612e-05, 'epoch': 1.16}                    
{'loss': 0.1707, 'grad_norm': 0.4722791016101837, 'learning_rate': 1.32079746261894e-05, 'epoch': 1.22}                      
{'loss': 0.1558, 'grad_norm': 0.6336535811424255, 'learning_rate': 1.2754870865428184e-05, 'epoch': 1.28}                    
{'loss': 0.1544, 'grad_norm': 0.709449291229248, 'learning_rate': 1.2301767104666971e-05, 'epoch': 1.34}                     
{'loss': 0.139, 'grad_norm': 0.7852916717529297, 'learning_rate': 1.1848663343905755e-05, 'epoch': 1.4}                      
{'loss': 0.1254, 'grad_norm': 0.7448189854621887, 'learning_rate': 1.1395559583144541e-05, 'epoch': 1.47}                    
{'loss': 0.1191, 'grad_norm': 0.5522488355636597, 'learning_rate': 1.0942455822383327e-05, 'epoch': 1.53}                    
{'loss': 0.1148, 'grad_norm': 0.45649582147598267, 'learning_rate': 1.0489352061622113e-05, 'epoch': 1.59}                   
{'loss': 0.105, 'grad_norm': 0.6231639385223389, 'learning_rate': 1.0036248300860897e-05, 'epoch': 1.65}                     
{'loss': 0.1055, 'grad_norm': 0.7918099164962769, 'learning_rate': 9.583144540099683e-06, 'epoch': 1.71}                     
{'loss': 0.099, 'grad_norm': 0.5217286944389343, 'learning_rate': 9.130040779338469e-06, 'epoch': 1.77}                      
{'loss': 0.0915, 'grad_norm': 0.4232180416584015, 'learning_rate': 8.676937018577255e-06, 'epoch': 1.83}                     
{'loss': 0.0915, 'grad_norm': 0.5594149827957153, 'learning_rate': 8.22383325781604e-06, 'epoch': 1.89}                      
{'loss': 0.0848, 'grad_norm': 0.5668164491653442, 'learning_rate': 7.770729497054826e-06, 'epoch': 1.95}                     
 67%|████████████████████████████████████████████████████▋                          | 3276/4914 [22:55:38<1:07:22,  2.47s/it]
 :
{'loss': 0.0783, 'grad_norm': 0.32559531927108765, 'learning_rate': 7.317625736293612e-06, 'epoch': 2.01}                    
{'loss': 0.0723, 'grad_norm': 0.4366374611854553, 'learning_rate': 6.864521975532397e-06, 'epoch': 2.08}                     
{'loss': 0.068, 'grad_norm': 0.3618405759334564, 'learning_rate': 6.411418214771183e-06, 'epoch': 2.14}                      
{'loss': 0.0687, 'grad_norm': 0.42383188009262085, 'learning_rate': 5.958314454009969e-06, 'epoch': 2.2}                     
{'loss': 0.0648, 'grad_norm': 0.3411434292793274, 'learning_rate': 5.505210693248755e-06, 'epoch': 2.26}                     
{'loss': 0.0674, 'grad_norm': 0.5947670936584473, 'learning_rate': 5.052106932487541e-06, 'epoch': 2.32}                     
{'loss': 0.0619, 'grad_norm': 0.41349029541015625, 'learning_rate': 4.599003171726326e-06, 'epoch': 2.38}                    
{'loss': 0.0628, 'grad_norm': 0.38769781589508057, 'learning_rate': 4.145899410965112e-06, 'epoch': 2.44}                    
{'loss': 0.0614, 'grad_norm': 0.30278515815734863, 'learning_rate': 3.692795650203897e-06, 'epoch': 2.5}                     
{'loss': 0.0612, 'grad_norm': 0.3795604109764099, 'learning_rate': 3.2396918894426825e-06, 'epoch': 2.56}                    
{'loss': 0.0569, 'grad_norm': 0.38001754879951477, 'learning_rate': 2.7865881286814683e-06, 'epoch': 2.63}                   
{'loss': 0.0551, 'grad_norm': 0.3927041292190552, 'learning_rate': 2.333484367920254e-06, 'epoch': 2.69}                     
{'loss': 0.0518, 'grad_norm': 0.3717590868473053, 'learning_rate': 1.8803806071590396e-06, 'epoch': 2.75}                    
{'loss': 0.054, 'grad_norm': 0.4690488278865814, 'learning_rate': 1.427276846397825e-06, 'epoch': 2.81}                      
{'loss': 0.0538, 'grad_norm': 0.3816925287246704, 'learning_rate': 9.74173085636611e-07, 'epoch': 2.87}                      
{'loss': 0.0548, 'grad_norm': 0.24697501957416534, 'learning_rate': 5.210693248753966e-07, 'epoch': 2.93}                    
 98%|███████████████████████████████████████████████████████████████████████████ 98%|███████████████████████████████████▏| 4802/4914 [23:59:45<05:19,  2.86s/it]{'loss': 0.0545, 'grad_norm': 0.27933555841445923, 'learning_rate': 6.796556411418215e-08, 'epoch': 2.99}
{'train_runtime': 86679.3606, 'train_samples_per_second': 0.907, 'train_steps_per_second': 0.057, 'train_loss': 0.32596833446572165, 'epoch': 3.0}
100%|████████████████████████████████████| 4914/4914 [24:04:39<00:00, 17.64s/it]
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:         train/epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:   train/global_step ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:     train/grad_norm ▃▄▂▃▄▆▅▅█▆▄▄▆▆▆▃▃▄▅▅▃▃▄▅▃▃▃▁▂▂▂▄▂▁▂▂▂▃▂▁
wandb: train/learning_rate ▂▄▅▇███▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▁▁▁
wandb:          train/loss █▇▇▆▅▄▃▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb:               total_flos 0
wandb:              train/epoch 3
wandb:        train/global_step 4914
wandb:          train/grad_norm 0.27934
wandb:      train/learning_rate 0.0
wandb:               train/loss 0.0545
wandb:               train_loss 0.32597
wandb:            train_runtime 86679.3606
wandb: train_samples_per_second 0.907
wandb:   train_steps_per_second 0.057
wandb: 
wandb: 🚀 View run custom-llama3-1b-pretraining at: https://wandb.ai/ggrover-farzi-na/llama3-training/runs/k28k270x
wandb: ⭐️ View project at: https://wandb.ai/ggrover-farzi-na/llama3-training
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
```