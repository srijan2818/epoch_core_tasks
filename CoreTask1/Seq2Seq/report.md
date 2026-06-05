# Code Refinement - Sequence to Sequence Model

The given dataset is : https://huggingface.co/datasets/google/code_x_glue_cc_code_refinement
- Everything is trained on the `small` variant.

## 1. Task

Creating sequence to sequence models which take buggy code as input and output fixed code.
- The goal is to understand different architectural choices, experiment with different seq2seq approaches and analyse and try to determine why certain approaches work or dont work well.
Following the requirement I tried these approaches and experiments:
1. RNN BiEncoder-Decoder with GRU cells and no attention
2. RNN BiEncoder-Decoder with GRU cells and Bahdanau attention
   1. Different teacher forcing ratios
3. Transformer Encoder-Decoder
   1. Different decoding methods
   2. Shared m=embeddings for encoder and decoder
   3. Different activations in FFN
   4. Testing on medium length dataset


## 2. Data analysis and distance metric

The given dataset has two splits : `small` and `medium` based on snippet length. Starting with `small` as shorter sequences show model failures faster and trying later on `medium` to see if the performance holds with more complexity or not.

- Since the identifiers are anonymised, theres no semantic context apart from the identifier numbers so the model doesnt learn patterns specific to different identifiers and should learn syntactic correctness.
- As a measure of distance between buggy and fixed code I used [Levenshtein edit distance](https://en.wikipedia.org/wiki/Levenshtein_distance) which measures how many token level edits are required to transform one string into another. It works by keeping track of how many insertion/delete/modification operations are required to transform one sequence to other starting from an empty string and perfoming comparisons incrementally.

- Mean train lengths (tokens by whitespace splitting)
  
|  |  |
|------|-------------|  
|source | 31.784 |
|target | 28.987 |
|edit distance | 7.233 |

The edit distance is significant in comparison to the source length showing that the edits arent just minor formatting changes and involves meaningful syntactic corrections.

## 3. Tokenisation

The dataset is already whitespace-split and anonymised. So `VAR_1` and `VAR_10` are different tokens in a whitespace vocab even though they share the same pattern. A whitespace split vocab treats them as independent symbols and with many training pairs and distinct VAR/METHOD/TYPE tokens they all could either get very similar embeddings with the cost of the model having to figure out their similarities from scratch or just produce more confusion because of low frequency.

Vocabulary:
- `429` tokens after whitespace splitting and `4` (`PAD`, `BOS`, `EOS` and `UNK`) special tokens.
- Sequence length = `50` , as $99\%$ of the snippets contain less than or equal to 50 tokens.
  
## 4. Testing Metrics
- Exact match - perfect prediction according to given fixed versions of code, low expected scores because of its extreme nature as one wrong token would mean absolute failure
<br><br>
- BLEU-4(Bilingual Evaluation Understudy): .<br> Tells how close wrong predictions are while treating code as natural language ignoring semnatics and syntax. It helps in tracking whether a model is converging towards the exact match or not but penalizes valid code heavily if variable names differ slightly<br>
$\text{Measure 1 to 4-gram overlap \% (with clipping)}  \rightarrow \text{Take weighted geometric mean of all the \%} \rightarrow \text{Apply length penalty} \rightarrow \text{Output score}\in [0,1]$ <br>
  $BLEU-4 = BP \cdot \exp\left(\sum_{n=1}^4 w_n \log p_n\right)$
<br><br>
- Edit similarity score: measures token-level distance required to transform the generated code into reference<br>
  $\text{Edit Similarity} = 1 -  \frac{Levenshtein(s_{gen},s_{ref})}{max(|s_{gen}|,|s_{ref}|)}$

* Out of these metrics BLEU and Edit similarity will be higher in most cases since a part of the code is fully copied from the original snippet in all cases and therefore arent very reliable metrics unless the difference is comparitively large.
  
## 5. Models

### 5.1 RNN (GRU Cells) Encoder-Decoder, No attention

- The RNN encoder is bidirectional and outputs both the concatenation of outputs of all layers reprsenting the compressed final states and the outputs at all time steps for the final layer, for both backward and forward, from which we take the final forward and backward states for all layers as the input state for the decoder. The decoder is unidirectional and takes the compressed encoder state as its initial hidden state, then generates the output sequence step by step using teacher forcing where each step's input is the ground truth token from the target.
- The encoder compresses the entire buggy code (~50) into a single vector of `d_model` floats. The bug could be at token 21 but till the time it processes  21-50 that information will get diluted or lost. The decoder has no way to lookback at individual source tokens since it only has compressed vector
- The recurrent compression process forces the model to compress exponentially growing information into a fixed space making its representation indistinguishable for long, complex inputs


## 5.2 Bahdanau Attention Model - Bidirectional encoder  + MLP Attention score
- The RNN encoder is bidirectional, same outputs as above. To address the fixed representation problem attention is used which lets the model focus on different parts of the input.  At each decoder step it decided which source parts (from all RNN states) are more important. 
- Given a decoder state $s_{t-1}$ and encoder states $h_1, h_2, \dotso, h_m$ scores, are calculated for how relevant source token $k$ is for target step $t$. Then attention weights $a_{k}^{(t)}$ are calculated through softmax. Then the final source context for decoder step $t$ is calculated as weighted sum with attenion weights.
- Attention score calculations could be dot-product, bilinear function (Luong model) and MLP (Bahdanau) - $e_{ti} = v^T \tanh(W_s s_{t-1} + W_h h_i^{enc})$ 
- The decoder now has information for all source tokens at any generation step. For an unchanged token at position 21, attention should peak at source position 10
- Its still recurrent and all recurrent models require $O(len(source) + len(target))$ per training step 
  
[BahdanauAttention](https://arxiv.org/abs/1409.0473) and [Reference](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)


## 5.3 Transformer Encoder - Decoder

- Transformer uses attention mechanism in encoder and decoder as well instead of recurrence or convolutions not just in their interactions
- The encoder now consists of all source tokens using their interactions with each other to update their representations N (blocks) times
- Decoder uses masked self attention for context from previous target tokens and cross-attention with queries from decoder states and keys, valeus from encoder states.
- For transformers since all tokens can be processed at once one trianing step required $O(1)$ steps

One `MultiHeadAttention` handles all three attention types via the `causal` arugment:
- Encoder self-attenton: no causal mask
- Decoder self-attention: causal mask
- Cross-attention: Q from decoder, K/V from encoder, no causal mask
  
- Run with both Pre-LN (norm before attention) and Post-LN (used in original paper). In Post-LN 
forward pass is :

```python
attn_out = Attention(x, x, x)
y = x + attn_out
z = LayerNorm(y)
```
So in backward pass $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} * \frac{\partial z}{\partial y}$ and for the skip connection path $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} * 1$ so within the gradient for the residual gets scaled affecting the connection

But in Pre-LN its
```python
x_norm = LayerNorm(x)
attn_out = Attention(x_norm, x_norm, x_norm)
y = x + attn_out
```
So in backward pass its $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} * 1$ allowing the skip connectinos gradient to flow freely \

## 6. Results and Experiments

### 6.1 Primary models results

All models have `d_model` = 128, `layers`=2, `n_enc\n_dec`=2 and `n_heads`=4 for transformer 
| Model | Exact | BLEU4 | Edit |
|-------|-------|------|--------|
RNN | 0.0300 |0.5324 |0.6582|
RNN Bahdanau |0.1500 |0.7346| 0.8031|
Transformer (Pre-LN) |0.1100| 0.7511| 0.8123|
Transformer (Post-LN)|0.0600| 0.7404| 0.7983|

Bahdanau attention performs well because it combines both context from encoder states and sequential states, however take comparitively larger time to train

Post-LN has $5\%$ lower exact accuracy and comparable BLEU and EditSimilarity scores, confirming that in Post-LN with the same configuration it isnt able to generalise and learn patterns as much as with Pre-LN.

### 6.2 Teacher Forcing Ablation for RNN with Attention

At default of TF=1.0, the decoder always receives the actual previous token, but at inference it only sees its own predictions.
- High TF should perform better since a single syntax error could lead to snowballing errors and the results confirm the same. 
- Even when only $20\%$ of the training steps use the model'own prediction, the errors compounds and since RNN has no mechanism to recover from a single incorrect token.
  
| Model | Exact | BLEU4 | Edit |
|-------|-------|------|--------|
RNN Bahdanau (tf=1.0) |0.1500 |0.7346| 0.8031|
RNN Bahdanau (tf=0.8) |0.0000 |0.0015| 0.0049|
RNN Bahdanau (tf=0.6) |0.0000 |0.0013| 0.0041|

### 6.3 Decoding strategies

Greedy decoding selects the highest-probability token at each step, which can miss better sequences if a slightly lower-probability token early on leads to a better result later. Beam seach maintains `k` candidate sequences (beams) at each step, expanding each by one token and keeping only top-k highest scoring beams - exploring multiple sequences adns eelcts final sequence with length-normalised score.

| Model | Beam Size | Temperature | Top-p | Exact | BLEU4 |
|-------|-----------|-------------|-------|-------|-------|
Transformer | 1 | 0.8 | 0.95 | 0.1100 | 0.7381 |
Transformer | 3 | 0.8| 0.95 |0.1100 | 0.7522 |
Transformer | 5 | 0.8| 0.95|0.1100 |0.7522 |
Transformer | 10 |0.8 | 0.95| 0.1100 |0.7522 |

Beam search improves BLEU score but doesnt increase exact match, showing that model's top-1 probability path is already concentrated on the correct token sequence for exact match / the probability distribution is sharp

### 6.4 Shared Embeddings (Transformer)

Tying embeddings reduces parameter count and force shared representation space between buggy and fixed code since they are formed from the same vocabulary. 

| Model | Exact | BLEU4 | Edit |
|-------|-------|------|--------|
Transformer (tie_embedding=False) |0.1100| 0.7511| 0.8123|
Transformer (tie_embedding=True) |0.1200| 0.7404| 0.8060|

The slight improvement shows that forcing encoder and decoder to use the same embedding space regularizes the model / prevents the decoder model to learn token representations different from encoder.

### 6.5 Activation Functions in FFN (Transformer)
1. SiLU (Sigmoid Linear Unit)  is a simple approximation of ReLU but without any discontinuity of the first derivative $\text{SiLU}(x) = x \cdot \sigma(x)$ 
2. SwiGLU (Swish-Gated Linear Unit) replaces the gating mechanism in GLU with the Swish activation function: $\text{SwiGLU}(x) = x_1 \cdot \text{Swish}(x_2)$
3. GEGLU (Gaussian Error Gated Linear Unit) is a variant of GLU that uses GELU instead of the sigmoid function: $\text{GEGLU}(x) = x_1 \cdot \text{GELU}(x_2)$
   
| Model | Activation | Hidden Dim | Exact | BLEU4 | Edit |
|-------|------------|------------|-------|-------|------|
Transformer (d_model=96, n_heads=4, tie_embedding=True) | SiLU | 128 | 0.1600 | 0.7621 | 0.8155 |
Transformer | SwiGLU | 256 | 0.0800 | 0.7459 | 0.8063 |
Transformer | GEGLU | 256 | 0.0900 | 0.7227 | 0.7948 |


GLU variants underperform SiLU despite 2x parameters - the gating mechanism overparameterizes this small dataset (46k samples), causing overfitting whereas SiLU allows small negative values to flow through and maintain gradient flow which is the likely reason it performs the best.

### 6.6 Testing on Medium Length Dataset

| Model | Dataset | Exact | BLEU4 | Edit |
|-------|---------|-------|-------|------|
Transformer (SiLU, d_model=96, n_heads=4, tie_embedding=True) | Small | 0.1600 | 0.7621 | 0.8155 |
Transformer (SiLU, d_model=96, n_heads=4, tie_embedding=True) | Medium | 0.0500 | 0.7984 | 0.7820 |

Exact match drops from 16% to 5% and edit similarity from 0.816 to 0.782, confirming the model struggles with longer bug patterns. BLEU goes up to 0.798 because it still catches local token overlaps, but fails at exact syntactic correctness over longer contexts. The model just overfits to the small dataset's length distribution.

## LLM / Other Resources usage:

- I have attached the links for the Bahdanau model references from which I learnt and implemented the model.
- I used LLM to understand BLEU metric and its shortcomings in code refinement tasks, and how gating mechanism works in SwiGLU and GEGLU