# Array Element Ranking

## 1. Task Definition

Given a sequence of integers, predict the relative sorted rank of each element.

```
Input  : [45, 12, 99, 31]
Ranks  : [2,  0,  3,  1]
```

This is a global reasoning task - determining any element's rank requires comparing it against every other element in the sequence - so an architecture is needed where global pairwise comparisons are possible - unmasked attention mechanism allows atleast the capability of representing global interactions than masked / autoregressive functions.

Normal sorting algorithms work sequentially, but if we train it with the objective of mimicking this method, the inference would also be autoregressive and it wont be able to modify previous outputs - so it has to produce all ranks in one forward pass.

The comparison function is $\text{sign}(x_i - x_j)$ and the attention mechanism computes

$\text{score}(i, j) = \frac{(W_q \cdot x_i) \cdot (W_k \cdot x_j)}{\sqrt{d_k}}$

it cant directly compute the difference of the projected vectors of inputs - so the model has to learn to approximate relational reasoning implicitly through its attention weights.



## 2. Dataset

- 10,000 data samples each with 10 non negative integers in the range [0,999] in first 10 columns and their corresponding ranks in the next 10 columns
- All values should have identical frequency distribution across its values range otherwise there's a possibility of model learning a pattern on its position which breaks the permutation invariancy requirement


## 3. Data Representation

Train / test / validation splits - standard 80 / 10 / 10.

Given the sample size is large enough, the mean of independently sampled values will converge to its actual mean so there's no need of stratification or other splitting methods.

Three representation strategies were explored:

1. **Categorical Projections** - Directly feed the raw integers into `nn.Embedding(1000, d_model)`. Treats each integer as a discrete symbol. The model has no inductive bias that 5 and 6 are mathematically close.

2. **Direct Normalisation + Projections** - Normalise the values within each sample and then pass into a linear projection. Normalisation can help because ranking remains same regardless of scale and within each sample because then the max and min are always 0 and 1 regardless of OOD samples. Tested both minmax and zscore.

3. **Positional-style Embedding using the value itself** - Normalise + apply sinusoidal encoding using the number itself rather than its position. This injects the mathematical relationship directly into the embedding via

$\text{PE}_{2i} = \sin(x \cdot \omega_i), \quad \text{PE}_{2i+1} = \cos(x \cdot \omega_i)$

For 3rd - I tried no normalisation, minmax and zscore - minmax range will be in $[0,1]$ whereas in zscore most values will lie in range $[-3,3]$ and with positional encoding with the negative values $\sin(x \cdot \omega_i)$ will give negative values which provides structural singal for values below sequence mean as well (in attention the dot product of members with flipped signs produces a large negative inner product)


## 4. Baselines

### 4.1 Training and Evaluation

- **Token accuracy** : Fraction of individual rank predictions correct across all positions
- **Sequence accuracy** : Fraction of sequences where every single rank is predicted correctly

These two metrics together give a fuller picture - a model can have high token accuracy while still failing on many sequences if it gets a few ranks consistently wrong.


### 4.2 MLP

A MLP over the embedded sequence - two hidden layers with ReLU activation and a final layer to map to sequence length. 
Tried multiple configurations by - changing capacity, learning rates and epochs but it doesnt learn the structure
This happens likely because MLP uses static weights. It cannot perform direct data-dependent pairwise comparisons across arbitrary positions

Results : Hidden layers: [64, 32] ; Epochs: 50 ; LR = 1e-3

Test tok accuracy: `0.3498` | seq accuracy: `0.0`

### 4.3 Bidirectional GRU

A bidirectional GRU processes the sequence and uses hidden states - the bidirectional structure at least allows each token to collect context from the full sequence before predicting its rank. 
The GRU updates its hidden states via sequential gating - calculating hidden states and updating with both historical context and relative magnitude shifts. 

Results:

Hidden: 128, Epochs: 30, LR: 1e-3
| Layers | Token Acc | Sequence Acc |
|--------|-----------|---------------|
| 1 | 0.812 | 0.161
| 2 | 0.9037 | 0.45
| 3 | 0.9265 | 0.564

Best result: Test tok accuracy: `0.9291` | seq accuracy: `0.595`

By concatenating both forward and backward states the linear head receives enough context to be able to predict extreme ends properly however it was unable to predict the ranks of 5th and 6th members properly showing that its unable to carry the distance metric without losing information across the sequence - or it didnt have the capability to track parallel pairwise distances

## 5. Encoder-Only Transformer

Unlike autoregressive models that process tokens sequentially, encoder-only transformers view the entire sequence simultaneously using bidirectional self-attention. This makes them effective for tasks requiring global relational understanding.

### 5.1 Architecture

The encoder stacks 2 transformer blocks and applies a linear classification head at each position to predict rank logits.

The `forward_with_weights` variant returns all intermediate attention matrices for visualisation - the softmax outputs before the V multiplication get stored.

**Config:** `d_model=64`, `n_heads=4`, `d_k=16`, `hidden=256`. 4 heads with `d_k=16` - each head works in a 16-dim subspace and can specialise independently.


## 6. Evaluation

### 6.1 Representation Ablations

Six embedder configurations tested on the same 4-layer transformer, 30 epochs, `lr=3e-4`:

| Embedder | tok_acc | seq_acc |
|---|---|---|
| NormProjectionEmbed_minmax | 0.9938 | 0.9440 |
| NormProjectionEmbed_zscore | 0.9954 | 0.9560 |
| CategoricalEmbed | 0.8193 | 0.2000 |
| PositionalEmbed_none | 0.9886 | 0.8990 |
| PositionalEmbed_minmax | 0.9919 | 0.9320 |
| PositionalEmbed_zscore | 0.9945 | 0.9480 |

- `CategoricalEmbed` lowest -  0.82 tok_acc and 0.20 seq_acc - the embedding table has no bias for numerical ordering so the model has to learn all pairwise relationships purely from supervision.
- The `PositionalEmbed` variants all lie slightly below their `NormProjection` counterparts, but even without normalisation the PostionalEmbed reaches high scores compared to CategoricalEmbed showing the reasoning was right.

### 6.2 Out-of-Distribution Testing

All OOD testing done with `NormProjectionEmbed_zscore`.

```
ext_clusters  | input:  [  5  12   9   3 992 985 997 980   8 994]
              | pred:   [1 3 2 0 7 5 9 5 2 8]
              | target: [1 4 3 0 7 6 9 5 2 8]

large_mag     | input:  [ 4000.  9000.  3000.  2000.  8000.  5000.  7000.  1000. 10000.  6000.]
              | pred:   [3 8 2 1 7 4 6 0 9 5]
              | target: [3 8 2 1 7 4 6 0 9 5]

all_ties      | input:  [5 5 5 5 5 5 5 5 5 5]
              | pred:   [8 8 8 8 8 8 8 8 8 8]
              | target: [0 0 0 0 0 0 0 0 0 0]

outlier       | input:  [  1   2   3   4   5   6   7   8   9 500]
              | pred:   [2 2 2 2 3 3 0 7 7 9]
              | target: [0 1 2 3 4 5 6 7 8 9]
```

- **ext_clusters**: fails only on the transition boundary between the two clusters - confuses ranks 4,5,6 within the tight [980-997] group. The rest is correct. The normalised projection compresses the intra-cluster gap so those tokens look nearly identical after embedding.
- **large_mag**: exact as expected - normalisation collapses any scale so 10x the training range doesn't matter.
- **all_ties**: predicts rank 8 for everything instead of 0. The all-equal case produces a flat attention distribution (every token is identical after embedding) and the model has no way to break the tie - it picks the mode it saw most frequently during training for ambiguous cases.
- **outlier**: badly fails. The single outlier at 500 gets ranked correctly at 9, but the tight cluster [1-9] gets compressed to near-zero after normalisation since the range is dominated by the 500. After embedding, those 9 values are almost indistinguishable and the model can't separate them.

### 6.3 Positional Encoding Ablation

Standard learned positional encoding added on top of value embeddings. Results with `NormProjectionEmbed_zscore`, 30 epochs:

Test tok_acc: `0.9689` | seq_acc: `0.7690`

OOD results with PE:
```
ext_clusters  | pred:   [2 2 2 1 7 6 7 6 2 8]   target: [1 4 3 0 7 6 9 5 2 8]
large_mag     | pred:   [9 6 4 7 3 2 5 0 8 1]    target: [9 6 4 7 3 2 5 0 8 1]
all_ties      | pred:   [0 0 0 0 0 0 0 0 0 0]    target: [0 0 0 0 0 0 0 0 0 0]
outlier       | pred:   [0 9 9 8 8 8 8 8 8 9]    target: [0 1 2 3 4 5 6 7 8 9]
```

PE drops tok_acc from 0.9954 to 0.9689 and seq_acc from 0.956 to 0.769. The rank of a number has nothing to do with where it appears in the input sequence - adding positional information introduces a pure noise signal the model partially learns to exploit on the training distribution but which doesn't generalise. The `all_ties` case - with PE each token has a unique position embedding so the model can distinguish them and outputs all 0s correctly, whereas without PE all tokens are identical and it outputs rank 8. 

### 6.4 Depth Ablation

`NormProjectionEmbed_zscore`, 30 epochs, `lr=3e-4`:

| Layers | tok_acc | seq_acc |
|--------|---------|---------|
| 1 | 0.7236 | 0.0540 |
| 2 | 0.9844 | 0.8740 |
| 4 | 0.9952 | 0.9550 |

The jump from 1 to 2 layers is large - seq_acc goes from 0.054 to 0.874. A single round of attention lets each token look at all others once, the model gets many individual ranks right but rarely gets the entire sequence correct. The second layer refines representation using the already contextualised outputs of 1. Layer 4 gives a further improvement but the gap is much smaller (0.874 - 0.955).

## LLM / Other resources usage : 

- I used LLM in this task to verify if the reasoning I came up with for the `PositionalEncoding` variants were justified or not after running the experiments and to write the animation code for the visualisation part.