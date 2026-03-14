# Task description
Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks), with any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

# Sources:
[Efficient Estimation of Word Representations in Vector Space]

[word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method]

[word2vec Parameter Learning Explained]

[Wikipedia]

[Wikipedia]:https://en.wikipedia.org/wiki/Word2vec

[word2vec Parameter Learning Explained]:https://arxiv.org/pdf/1411.2738

[word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method]:https://arxiv.org/pdf/1402.3722

[Efficient Estimation of Word Representations in Vector Space]:https://arxiv.org/pdf/1301.3781

# Methodology
## Datasets for training and testing
[IMDB Reviews]:https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
[Amazon Reviews]:https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews

I have used [IMDB Reviews] dataset (actually I have downloaded it from HuggingFace). It contains around 50k reviews, 25k for tests and 25k for training.
However it is large, so I have chosen only part of this dataset.

Another interesting, not yet tested - [Amazon Reviews].

## word2vec overview

### Continuous bag-of-words (CBOW)
I will use the notation from [Wikipedia]:
1) C - corpus of words (sequence of words),
2) V - set of words in the corpus C, our dictionary,
3) D - cardinality of set V,
4) N - relative locations of nearby words, e.g. {-2,-1,1,2}; 0 does not belong to N,
5) d - dimension of hidden layer (latent space).

Intuitively we want to predict given word in sequence according to it's nearby words, 
to be more precise we define neighbour words of word $w_i$ as $w_{i+j} \forall_{j \in N}$.
Our objective is to maximize:

$\sum_{i \in |C|} \ln(P(w_i|w_{i+j} \forall_{j \in N}))$

We use linear-linear-softmax architecture. Our input is $x \in R^{D \times 1}$, which encodes histogram
of words in neighbourhood of given word. We have 2 learnable linear projection matrices:
$V \in R^{d \times D}$ and $V' \in R^{D \times d}$. Output probabilities are
$\text{softmax}(V'Vx) \in [0,1]^{D \times 1}.$

Our objective is equivalent to minimizing

$L = -\sum_{i \in |C|} \ln(P(w_i|w_{i+j} \forall_{j \in N})) = 
-\sum_{i \in |C|} \ln( \frac{e^{v'_{w_i} (V z_i)}}{\sum_{w' \in V} e^{v'_{w'} (V z_i)}}) = 
-\sum_{i \in |C|} \ln( \frac{e^{v'_{w_i} (\sum_{j \in N} v_{i+j})}}{\sum_{w' \in V} e^{v'_{w'} (\sum_{j \in N} v_{i+j})}}),$

where $z_i \in R^{D \times 1}$ is histogram of nearby words of word $w_i$. This is the same as cross entropy loss defined in [Wikipedia]. However in reality I have used mean over words in the corpus.

Note that this implementation is inefficient, as dictionary can be huge. Better implementation could use
Negative Sampling (that would probably require changing objective) or Hoffman codes.

Note that if $N$ (indices set of relative locations of nearby words) is small, and dictionary size 
$D$ is huge, then majority of entries in input (histogram of words in the neighbourhood) are zeros.
Naive computation has complexity of O(Dd) time. Note that forward pass could omit some computation 
and just set output of the first layer as

$y = Vx = \sum_{j \in N} v_{w_{i+j}}.$

This costs $O(|N| D)$.

However as probably each vector $v_{i+j}$ will not have a lot of zeros, then computing output of second layer

$u = V'y$

will require $O(Dd)$.
After that softmax has complexity $O(D)$. As we can see overall complexity of forward pass is $O(Dd)$, where second linear layer
enforces such high cost. We can omit optimization described earlier regarding first linear projection, as it only reduces constant, but not complexity.

### Skip-gram with Negative Sampling
Architecture is the same as in case of CBOW.

Let's denote pair of word and one word from it's context as (w,c), then

$P(v_c'|v_w) = \text{sigmoid}(v_c \cdot v_w')$

Let's denote $\text{sigmoid} = \sigma$. 

We use Negative Sampling (otherwise model could learn trivial representations). Let's denote set of positive pairs as $D$
and set of negative pairs as $D'$. Then our loss is

$L = - \sum_{(w,c) \in D}\ln(\sigma(v_c' \cdot v_w)) - \sum_{(w,n) \in D'} \ln(\sigma(v_n' \cdot v_w))$

Actually for each positive pair $(w,c)$ we sample k negative pairs $(w,c_1), ..., (w,c_k)$. We actually do not check if they are wrong, we just sample them from unigram distribution raise to power 3/4  (see [word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method]). In reality for each pair we compute loss and compute mean over number of positive pairs in the batch.

## Important remarks
I have used torch only for tensorboard.

## Running the code
Examples of training CBOW:
```
python3 main.py  --min_occurences=5 --r --is_CBOW --ds=500
```

Example of training Skip-Gram with Negative Sampling:
```
python3 main.py --min_occurences=5 --r --ds=5000
```

Some results are in `out/cbow` and `out/skip_gram` folders.