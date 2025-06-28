

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
(Pdb) config
LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 2816,
  "max_position_embeddings": 2048,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 16,
  "num_hidden_layers": 12,
  "num_key_value_heads": 8,
  "pad_token_id": 128004,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.52.4",
  "use_cache": true,
  "vocab_size": 128256
}

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 1024, padding_idx=128004)
    (layers): ModuleList(
      (0-11): 12 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (k_proj): Linear(in_features=1024, out_features=512, bias=False)
          (v_proj): Linear(in_features=1024, out_features=512, bias=False)
          (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=1024, out_features=2816, bias=False)
          (up_proj): Linear(in_features=1024, out_features=2816, bias=False)
          (down_proj): Linear(in_features=2816, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((1024,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=128256, bias=False)
)
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
LlamaForCausalLM                              --                        --
├─LlamaModel: 1-1                             --                        --
│    └─Embedding: 2-1                         [4, 512, 1024]            131,334,144
│    └─LlamaRotaryEmbedding: 2-2              [1, 512, 64]              --
│    └─ModuleList: 2-3                        --                        --
│    │    └─LlamaDecoderLayer: 3-1            [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-2            [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-3            [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-4            [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-5            [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-6            [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-7            [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-8            [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-9            [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-10           [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-11           [4, 512, 1024]            11,798,528
│    │    └─LlamaDecoderLayer: 3-12           [4, 512, 1024]            11,798,528
│    └─LlamaRMSNorm: 2-4                      [4, 512, 1024]            1,024
├─Linear: 1-2                                 [4, 512, 128256]          131,334,144
===============================================================================================
Total params: 404,251,648
Trainable params: 404,251,648
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 1.62
===============================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 4450.16
Params size (MB): 1617.01
Estimated Total Size (MB): 6067.18
===============================================================================================
wandb: ⭐️ View project at https://wandb.ai/ggrover-farzi-na/llama3-training
wandb: 🚀 View run at https://wandb.ai/ggrover-farzi-na/llama3-training/runs/rta7agtg
INFO:__main__:Starting training...
  0%|                                                 | 0/42495 [00:00<?, ?it/s]
{'loss': 11.4834, 'grad_norm': 3.700112819671631, 'learning_rate': 3.96e-06, 'epoch': 0.01}
{'loss': 9.7267, 'grad_norm': 2.1536293029785156, 'learning_rate': 7.960000000000002e-06, 'epoch': 0.01}
{'loss': 8.6046, 'grad_norm': 1.524652361869812, 'learning_rate': 1.196e-05, 'epoch': 0.02}
{'loss': 7.4222, 'grad_norm': 1.5482126474380493, 'learning_rate': 1.5960000000000003e-05, 'epoch': 0.03}
{'loss': 6.8732, 'grad_norm': 1.3972337245941162, 'learning_rate': 1.9960000000000002e-05, 'epoch': 0.04}
{'eval_loss': 6.735888957977295, 'eval_runtime': 555.3573, 'eval_samples_per_second': 12.623, 'eval_steps_per_second': 3.157, 'epoch': 0.04}                    
  1%|▍                                   | 500/42495 [41:36<45:34:14,  3.91s/itINFO:__main__:███████████████████████████████| 1753/1753 [09:15<00:00,  2.86it/s]
🔍 Prompt Evaluation @ step 500
INFO:__main__:📝 Prompt: Once upon a time
INFO:__main__:🧠 Model Output: Once upon a time the 2, 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010, 201
INFO:__main__:--------------------------------------------------
INFO:__main__:📝 Prompt: The capital of France is
INFO:__main__:🧠 Model Output: The capital of France is, and the 1 @,@ 1 @,@ 1 @,@ 1 @,@ 1 @,@ 1 @,@ 1 @,@ 1 @,@ 1 @,@ 1 @,@ 1 @,@ 1 @
INFO:__main__:--------------------------------------------------
{'loss': 6.6071, 'grad_norm': 1.1762800216674805, 'learning_rate': 1.9952851529944042e-05, 'epoch': 0.04}                                                       
{'loss': 6.4478, 'grad_norm': 1.2886886596679688, 'learning_rate': 1.9905226812715802e-05, 'epoch': 0.05}
{'loss': 6.3273, 'grad_norm': 1.3849259614944458, 'learning_rate': 1.985760209548756e-05, 'epoch': 0.06}
{'loss': 6.1898, 'grad_norm': 1.5535489320755005, 'learning_rate': 1.980997737825932e-05, 'epoch': 0.06}
{'loss': 6.121, 'grad_norm': 1.652300477027893, 'learning_rate': 1.9762352661031076e-05, 'epoch': 0.07}
  2%|▊                                | 1000/42495 [1:14:11<44:50:22,  3.89s/it]
{'eval_loss': 6.080770969390869, 'eval_runtime': 554.8554, 'eval_samples_per_second': 12.634, 'eval_steps_per_second': 3.159, 'epoch': 0.07}                    
  2%|▊                                | 1000/42495 [1:23:26<44:50:22,  3.89s/itINFO:__main__:███████████████████████████████| 1753/1753 [09:14<00:00,  3.16it/s]
🔍 Prompt Evaluation @ step 1000
INFO:__main__:📝 Prompt: Once upon a time
INFO:__main__:🧠 Model Output: Once upon a time, and the 2010s, the 2010s, and the 2010s, and the 2010s, and the 2010s, and the 2010s, and the 2010s, and
INFO:__main__:--------------------------------------------------
INFO:__main__:📝 Prompt: The capital of France is
INFO:__main__:🧠 Model Output: The capital of France is, and the " ". 
 = = = = = 
 = = = = = = 
 The first time of the 2010s, the 2010s, the 2010s, the 2010s, and the
INFO:__main__:--------------------------------------------------
{'loss': 6.031, 'grad_norm': 1.2775781154632568, 'learning_rate': 1.9714727943802836e-05, 'epoch': 0.08}                                                        
{'loss': 5.9556, 'grad_norm': 1.7863184213638306, 'learning_rate': 1.9667103226574593e-05, 'epoch': 0.08}
{'loss': 5.9054, 'grad_norm': 1.6831647157669067, 'learning_rate': 1.9619478509346353e-05, 'epoch': 0.09}
{'loss': 5.8408, 'grad_norm': 1.8831902742385864, 'learning_rate': 1.957185379211811e-05, 'epoch': 0.1}
{'loss': 5.7951, 'grad_norm': 1.8510370254516602, 'learning_rate': 1.952422907488987e-05, 'epoch': 0.11}
  4%|█▏                               | 1500/42495 [1:55:59<44:24:47,  3.90s/it]
{'eval_loss': 5.772325038909912, 'eval_runtime': 553.6469, 'eval_samples_per_second': 12.661, 'eval_steps_per_second': 3.166, 'epoch': 0.11}                    
  :
  ::
🔍 Prompt Evaluation @ step 7000
INFO:__main__:📝 Prompt: Once upon a time
INFO:__main__:🧠 Model Output: Once upon a time, and the film was not a " very good ". 
 = = = = = 2000 = = = = = 
 In 2008, the film was released on DVD in the United States on 9 August 2008.
INFO:__main__:--------------------------------------------------
INFO:__main__:📝 Prompt: The capital of France is
INFO:__main__:🧠 Model Output: The capital of France is the most important and most important of the world. The most important of the world's most important works are the most important of the world's most important works, the most important of the world's most important and most important, the most
INFO:__main__:--------------------------------------------------
{'loss': 4.6574, 'grad_norm': 1.9189836978912354, 'learning_rate': 1.685724491010835e-05, 'epoch': 0.5}                                                         
{'loss': 4.6761, 'grad_norm': 1.8837392330169678, 'learning_rate': 1.6809620192880106e-05, 'epoch': 0.51}
{'loss': 4.6667, 'grad_norm': 1.7942728996276855, 'learning_rate': 1.6761995475651866e-05, 'epoch': 0.52}
{'loss': 4.6606, 'grad_norm': 1.8920258283615112, 'learning_rate': 1.6714370758423623e-05, 'epoch': 0.52}
{'loss': 4.6456, 'grad_norm': 2.4267821311950684, 
:
:
 21%|██████▊                         | 9000/42495 [12:22:27<36:02:32,  3.87s/it]
  warnings.warn(warn_msg)
{'eval_loss': 4.507755279541016, 'eval_runtime': 551.6732, 'eval_samples_per_second': 12.707, 'eval_steps_per_second': 3.178, 'epoch': 0.64}                    
 21%|██████▊                         | 9000/42495 [12:31:39<36:02:32,  3.87s/itINFO:__main__:███████████████████████████████| 1753/1753 [09:11<00:00,  3.18it/s]
🔍 Prompt Evaluation @ step 9000
INFO:__main__:📝 Prompt: Once upon a time
INFO:__main__:🧠 Model Output: Once upon a time of the 20th century, and the 13th century, the 13th century, the 13th century, the 13th century, the 13th century, the 13th century, the 13th century,
INFO:__main__:--------------------------------------------------
INFO:__main__:📝 Prompt: The capital of France is
INFO:__main__:🧠 Model Output: The capital of France is the most important and most important and popular culture in the world. 
 = = = = 2000 – 2000 = = = = 
 In 2005, the United States was the first of the 2000s to be the
INFO:__main__:--------------------------------------------------
{'loss': 4.4937, 'grad_norm': 1.8304232358932495, 'learning_rate': 1.590475056554352e-05, 'epoch': 0.64}                                                        
{'loss': 4.501, 'grad_norm': 1.8170983791351318, 'learning_rate': 1.5857125848315277e-05, 'epoch': 0.65}
{'loss': 4.4817, 'grad_norm': 1.977964997291565, 'learning_rate': 1.5809501131087033e-05, 'epoch': 0.66}
{'loss': 4.4898, 'grad_norm': 2.1780858039855957, 'learning_rate': 1.5761876413858794e-05, 'epoch': 0.66}
{'loss': 4.5005, 'grad_norm': 1.8188778162002563, 'learning_rate': 1.5714251696630554e-05, 'epoch': 0.67}
 22%|███████▏                        | 9500/42495 [13:03:54<35:21:44,  3.86s/it]
  warnings.warn(warn_msg)
{'eval_loss': 4.468681335449219, 'eval_runtime': 548.6878, 'eval_samples_per_second': 12.776, 'eval_steps_per_second': 3.195, 'epoch': 0.67}                    
 ::
 ::
 [13:54:24<34:45:55,  3.85s/itINFO:__main__:███████████████████████████████| 1753/1753 [09:09<00:00,  3.19it/s]
🔍 Prompt Evaluation @ step 10000
INFO:__main__:📝 Prompt: Once upon a time
INFO:__main__:🧠 Model Output: Once upon a time of the film, and the film's " a little bit of a lot of the film ". 
 = = = = 2000 – 2001 = = = = 
 In 2001, the film was released on DVD and
INFO:__main__:--------------------------------------------------
INFO:__main__:📝 Prompt: The capital of France is
INFO:__main__:🧠 Model Output: The capital of France is the most important and popular culture of the world. 
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
INFO:__main__:--------------------------------------------------
{'loss': 4.4045, 'grad_norm': 1.8270361423492432, 'learning_rate': 1.5428503393261102e-05, 'epoch': 0.71}                                                       
{'loss': 4.4327, 'grad_norm': 1.848037600517273, 'learning_rate': 1.5380878676032862e-05, 'epoch': 0.72}
{'loss': 4.4367, 'grad_norm': 1.8245179653167725, 'learning_rate': 1.5333253958804622e-05, 'epoch': 0.73}
{'loss': 4.396, 'grad_norm': 1.9088610410690308, 'learning_rate': 1.528562924157638e-05, 'epoch': 0.73}
 25%|███████▌    
```