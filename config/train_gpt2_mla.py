# config for training GPT-2 (124M) with Multi-Latent Attention
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train_mla.py config/train_gpt2_mla.py

wandb_log = True
wandb_project = 'gpt2mla'
wandb_run_name = 'gpt2-124M-mla'

# these make the total batch size be ~0.5M
# 6 batch size * 1024 block size * 10 gradaccum * 8 GPUs = 491,520
batch_size = 6
block_size = 1024
gradient_accumulation_steps = 10 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# model parameters
n_layer = 12
n_head = 12
n_embd = 768
n_latent = 384  # latent dimension (half of embedding dimension)
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False    # do we use bias inside LayerNorm and Linear layers?

# optimizer parameters
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
min_lr = 6e-5

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16'
compile = False # turned off to avoid OOM during compilation 