# Poetry Generation

## 1. Task Definition

Generate coherent poetry given a prompt - the model is trained to predict next token autoregressively, learning the statistical structure of poetic language.

The dataset contains poems with special formatting characters (line breaks, punctuation, diacritics) that need to be preserved - poetry structure matters.

## 2. Dataset

- Original dataset had LaTeX-style encoding for special characters like `{oe}`, `{"e}`, `{i}{_o}` etc and also use CRLF (`\r\n`) which i changed to just `\n`.
- 567k raw characters, 83 unique characters before cleaning
- After replacing special sequences, final length 552,733 characters, unique characters reduced to 79 (including line breaks and punctuation)

```
Special chars mapping:
{oe} -> œ
{"e} -> ë
{.} -> ye
{i}{_o} -> io
{_o} -> o
{i} -> i
```

Data split: 80/10/10 train/val/test at poem level, not character level. Each poem treated as a complete unit.

## 3. Tokenisation

Word-level tokenisation with special tokens:

- `<BOS>`, `<EOS>`, `<PAD>`, `<EOL>` tokens for sequence boundaries and padding
- Vocabulary size: `13,107` tokens
- Words separated by spaces, punctuation kept as separate tokens, newlines as `<EOL>`

The tokeniser preserves line breaks explicitly so the model can learn stanza structure.

## 4. Baselines

### 4.1 Bigram Model

Simple count-based language model - predicts next token based only on previous token.

PPL: `1552.86`

Generated text from random token:

```
the thing to pine,the nunnery beaches
For lo the thought
Seeing iron,yea,behind
```

The output lacks coherence but shows basic English word co-occurrence patterns. No long-range structure.

### 4.2 MLP Baseline

Multi-layer perceptron over embedded sequence - processes fixed window of tokens without recurrence or attention.

Hidden layers: [256,128], d_model=256, LR=1e-3

Test ppl: `499.29`

Performs better than the bigram model but remains limited - no connectivity between predictions as each position's output depends only on its own embedding.

### 4.3 GRU Baseline

3-layer bidirectional GRU, d_model=512, hidden=128, 100 epochs

Test ppl: `322.92`

Significant improvement over MLP - the hidden state propagates information forward, enabling the model to maintain some continuity. Bidirectional structure allows each token to access both past and future contexts during training.

Generated from prompt "And the":

```
And the woods within my heart,
Her death for thy powr
Of Olive summers night before,
Love,they all my show,and I saw the green heart;
Like fair love-wind my own.
```

Some lines follow poetic structure. The model learned stanza breaks, line lengths, and occasional rhyming patterns.

## 5. Decoder-Only Transformer

Causal transformer with masked self-attention - each position attends only to previous positions.

**Config tested:**
- d_model: 256
- n_heads: 4
- n_layers: 2
- hidden: 512
- Seq len: 128
- Dropout: 0.1
- Learned positional encoding

Training: 30 epochs, LR=1e-3, weight_decay=0.01

Test ppl: `420.87`

Multiple configurations (more epochs, different layer counts, varied regularization parameters) were attempted but did not yield improvements within the available experimentation time. The reported configuration represents the best result obtained under the constraints.

## 6. Decoding Strategies

Temperature scaling, top-p sampling, and repetition penalty were evaluated.

### Greedy (temperature=0):
```
And the wanderer the Image of the world,
And
The nombers.
And the riper
And the world,
The primitias the world
```
Produces repetitive sequences with low diversity.

### Temperature 0.7:
```
And the wanderer every with water of earthy all the basest strawberries tree
And and weeds the commanding
And we the same,
But
Brings Thames thou of one as the basest he the outward,
```
Increased diversity at the cost of coherence in some positions.

### Top-p 0.8 + Temperature 0.8:
```
And the sand"Canto LXXXI,
And the bills,
Herald the vestal and from the latest is to the front and the world
We had the ground it is a of the prayse the place of steel blaze the eagle
And in my eyes.
```
Produces more structured poetic fragments with appropriate capitalization and quotation patterns.

### Top-p 0.9 + rep penalty 1.3 + Temperature 0.8:
```
And the Mermaids"Canto LXXXI,
His bones and that University run which
He people I of the sea-borderers self her against,repeats to it all alone.
My high our fools and be one with thee this O,or my own my request it hath all these art.
And we thoughts in you decayed shadow of the waves hast an and then,
```
Most coherent output among the strategies tested. Repetition penalty reduces token recycling, encouraging model to explore more possible directions. Grammatical inconsistencies remain, but the output approximates poetic structure.
  


