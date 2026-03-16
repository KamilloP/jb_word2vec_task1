import numpy as np
import re
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
import datetime

###########################################################
########################## Utils ##########################
###########################################################

def numpy_dataloader(dataset, batch_size=32):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        yield batch

def softmax(x : np.ndarray, axis=0) -> np.ndarray:
    """
    Expects x of shape (d_1, d_2, ..., d_n), returns softmax over dimension axis.
    """
    assert -len(x.shape) <= axis and axis < len(x.shape), f"Argument axis is out of range (axis={axis}, x.shape={x.shape})"
    exp = np.exp(x - x.max(axis=axis, keepdims=True)) # Subtraction for numerical stability
    return exp / exp.sum(axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_softmax(x: np.ndarray, axis=-1):
    """
    Expects x of shape (d_1, d_2, ..., d_n), returns softmax over dimension axis.
    """
    x_shift = x - x.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(x_shift).sum(axis=axis, keepdims=True))
    return x_shift - logsumexp

def textToWords(text, dictionary=None):
    """
    Gathers words from the text, puts them to lowercase.
    If dictionary is collection such as dict, then checks if words are present in the
    dictionary and and add them to the result list.
    If dictionary=None, then add all words to the result list.
    """
    l = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if dictionary is None:
        return l
    res = []
    for e in l:
        if e in dictionary:
            res.append(e)
    return res

def nearbyWords(words_sequence, neighbourhood_indices, i, D):
    """
    Returns a histogram of context words around words_sequence[i], shape of result is (D,1).
    words_sequence is ndarray containing sequence of indices of the words from the text in the dictionary.
    Assumes that neighbourhood_indices is sorted ndarray.
    Assumes that each word index belongs to the dictionary.
    """
    valid = neighbourhood_indices[(neighbourhood_indices + i >= 0) & (neighbourhood_indices + i < len(words_sequence))]
    z = np.zeros((D,1), dtype=float)
    np.add.at(z[:,0], words_sequence[i+valid], 1) # add.at is important as repetitions may appear. 
    return z

def createCorpus(data, dictionary):
    """
    Creates corpus for dataset. Corpus is list of ndarrays, corpus[i] contains
    data[i] sequence mapped to indices in dictionary.
    """
    return [
        np.array([dictionary[w] for w in textToWords(text, dictionary=dictionary)], dtype=np.int32)
        for text in data
    ]

def createDictionaryAndCounter(data, min_value=1., max_value=np.inf):
    """
    This method creates dictionary based on words in data.
    It inserts words to dictionary only if number of word occurrences in data is not
    too big or too small. On default it assumes that every positive number of occurrences is sufficient.
    The reason for minimal and maximal threshold of occurrences is that we want dictionary to not
    be too large while too frequent words do not bring a lot of information. 
    """
    c = Counter()
    for text in data:
        for w in textToWords(text):
            c[w] += 1
    for e in list(c):
        if c[e] < min_value or c[e] > max_value:
            del c[e]
    dictionary = {k: i for i, k in enumerate(sorted(c))}
    counter = np.zeros(len(dictionary), dtype=float)
    for word, count in c.items():
        counter[dictionary[word]] = count
    return dictionary, counter

###########################################################
####################### Naive CBOW ########################
###########################################################

class NaiveCBOW():
    """
    This is naive implementation of Continuous bag-of-words (CBOW) as it uses whole histogram context vectors.
    More efficient implementation would use something like hierarchical softmax. 
    Forward produces probability using softmax, for good
    """
    def __init__(self, D, d, dictionary, neighbourhood_indices, seed=42):
        np.random.seed(seed)
        self.dictionary = dictionary
        self.V = np.random.randn(d,D) / np.sqrt(D)
        self.V_ = np.random.randn(D,d) / np.sqrt(d)
        self.D = D # cardinality of set V.
        self.d = d # dimension of hidden layer (latent space).
        self.neighbourhood_indices = neighbourhood_indices

    def forward(self, x):
        """
        Expects x.shape = (..., D, 1)
        """
        x = self.V @ x
        x = self.V_ @ x
        return softmax(x, axis=-2)
    
    def loss(self, x, w_i):
        """
        Arguments:
        -- x - batch of histogram of words in the neighbourhood of corresponding words in w_i. Note that x.shape = (B, D, 1)
        -- w_i - indices in the dictionary of the word we compute loss for. w_i.shape == (B) 
        """
        assert len(x.shape) == 3
        assert x.shape[2] == 1
        assert w_i.shape == (x.shape[0],)
        g = self.V_ @ (self.V @ x) 
        # This could be optimized (same complexity, but smaller constant if we omit 0 in z), but on the other hand we cannot
        # do this easily with batch, and using np will probably result with smaller constant than this optimization. However
        # problem of huge memory overhead remains when we use this function.
        logp = log_softmax(g, axis=-2) # Better for numerical stability than log(softmax(...)).
        return -logp[np.arange(x.shape[0]), w_i, 0].mean()
    
    def gradientLoss(self, x, w_i):
        # x: (B,D,1)
        B,_,_ = x.shape
        z = self.V @ x # z: (B,d,1)
        g = self.V_ @ z # g: (B,D,1)
        # M = softmax(g, axis=-2)[np.arange(x.shape[0]), w_i, 0] # M: (B)
        # BL = -np.log(M) # BL: (B)
        #L = BL.sum()
        # assert L == self.loss(x, w_i) # It probably would not work, we should use some precision, not ==.
        # Below we compute loss in batch of losses BL separately. We will add gradients later. 
        dBL_dg = softmax(g, axis=-2)
        dBL_dg[np.arange(B), w_i, 0] -= 1. # (B, D, 1)
        # dg_dV_{a,b} = [0, 0, ..., z_b, 0, 0, ..., 0], where z_b is in a-th entry, remember that b \in {0, ..., d-1}.
        dL_dV_ = np.mean(dBL_dg * z.transpose(0, 2, 1), axis=0)
        dBL_dz = np.matmul(self.V_.T, dBL_dg) # Because dg_dz = self.V_. 
        # We use chain rule. dBL_dz.shape == (B, d, 1). Analogically as in case of dL_dV_:
        dL_dV = np.mean(dBL_dz * x.transpose(0,2,1), axis=0)
        return dL_dV_, dL_dV
    
    def __schemeForCorpus(self, goal_func, init_func, add_func, mul_func, corpus, computeUsingBatch=True):
        """
        Auxiliary function for computing mean objective functions (e.g. loss or gradients) over a corpus.
        Arguments:
        -- corpus - list of lists, corpus[i][j] contains j-th word in i-th text. Assumes that each word is in dictionary.
        """
        target = init_func() # cross entropy loss
        nr_of_samples = 0

        for list_ in corpus:
            nr_of_samples += len(list_)
            temp_target = init_func()
            if computeUsingBatch:
                z = np.zeros((len(list_), self.D, 1)) # Can be huge
                for i in range(len(list_)):
                    z[i] = nearbyWords(list_, self.neighbourhood_indices, i, self.D)
                temp_target = goal_func(z, np.arange(len(list_)))
                temp_target = mul_func(temp_target, len(list_))
            else:
                # Compute 1 by 1
                for i in range(len(list_)):
                    z = nearbyWords(list_, self.neighbourhood_indices, i, self.D)
                    val = goal_func(np.expand_dims(z, axis=0), np.arange(1))
                    temp_target = add_func(temp_target, val)
            target = add_func(target, temp_target)
        target = mul_func(target, 1./nr_of_samples)
        return target
    
    def lossForCorpus(self, corpus, computeUsingBatch=True):
        """
        Returns loss (single number).
        """
        return self.__schemeForCorpus(
            self.loss,
            (lambda: 0.),
            (lambda a, b: a+b), 
            (lambda a, d: a*d), 
            corpus, 
            computeUsingBatch=computeUsingBatch
        )
    
    def gradientForCorpus(self, corpus, computeUsingBatch=True):
        """
        Returns dL_dV_, dL_dV.
        """
        return self.__schemeForCorpus(
            self.gradientLoss,
            (lambda: (np.zeros_like(self.V_), np.zeros_like(self.V))),
            (lambda a, b: (a[0]+b[0], a[1]+b[1])),
            (lambda a, d: (a[0]*d, a[1]*d)),
            corpus,
            computeUsingBatch=computeUsingBatch
        )
    
    def train(self, train_dataset, test_dataset, epochs, batch_size, lr, logdir="out/cbow", computeUsingBatch=True):
        """
        Training loop. We do not use validation dataset.
        """
        writer = SummaryWriter(logdir + "/" + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        list_of_corpuses = []
        for data in numpy_dataloader(train_dataset, batch_size):
            list_of_corpuses.append(createCorpus(data, self.dictionary))
        test_corpus = createCorpus(test_dataset, self.dictionary)
        for e in range(epochs):
            print(f"Epoch {e}:")
            loss = 0
            for i, corpus in enumerate(list_of_corpuses):
                loss = self.lossForCorpus(corpus, computeUsingBatch)
                dL_dV_, dL_dV = self.gradientForCorpus(corpus, computeUsingBatch)
                self.V_ -= lr*dL_dV_
                self.V -= lr*dL_dV
                writer.add_scalar("train/loss", loss, len(list_of_corpuses)*e+i+1)
                print(f"Training loss at step {i}: {loss}")
            test_loss = self.lossForCorpus(test_corpus, computeUsingBatch=computeUsingBatch)
            print(f"Testing loss at epoch {e}: {test_loss}")
            writer.add_scalar("test/loss", loss, e*len(list_of_corpuses))
        writer.close()

##########################################################
######### Skip-gram model with Negative Sampling #########
##########################################################

class SkipGram():
    """
    Skip-gram with Negative Sampling. It uses mean over positive pairs in the batch.
    For each pair we sample k negative samples. Window sets maximal context. Context is dynamic by default,
    but can be set otherwise. We do not check if negative samples are truly negative, we just sample the 
    context word according to unigram distribution raised to power 3/4. For this we use "count" argument -
    number of word occurrences in the dictionary.
    """
    def __init__(self, D, d, count, window=10, k=10, seed=42):
        np.random.seed(seed)
        self.D = D # number of words in the dictionary
        self.d = d # embedding dimension
        self.k = k # number of negative samples per positive pair
        self.window = window
        # Note that in the definition of negative sampling in paper https://arxiv.org/pdf/1402.3722
        # they take unigram distribution and then raise it to 3/4, but this may not sum up to 1.
        # So we divide it here by sum to make proper distribution.
        count = count**(3/4)
        self.unigram_prob = count / count.sum()
        self.unigram_prob /= self.unigram_prob.sum()

        self.V = np.random.randn(d, D) / np.sqrt(D)
        self.V_ = np.random.randn(D, d) / np.sqrt(d)
    
    def positive_samples(self, sequence_id, i, dynamic_window=True):
        """
        Samples positive pairs. If dynamic_window=True, then context is dynamic
        (window can be smaller than window). Otherwise window is the same. 
        """
        R = self.window # Context length
        if dynamic_window:
            R = np.random.randint(1, self.window+1)
        c = []
        left = sequence_id[max(i-R,0):i]
        right = sequence_id[i+1: min(i+R+1, len(sequence_id))]
        if len(left) + len(right) == 0:
            return np.array([],dtype=int), np.array([],dtype=int)
        c = np.concatenate((left, right))
        w = np.full_like(c, sequence_id[i])

        return w,c
    
    def sample_negatives(self, nr_of_positive_samples):
        c = np.random.choice(self.D, nr_of_positive_samples*self.k, p=self.unigram_prob)
        return c
    
    def forward(self, w, c):
        vw = self.V[:, w].reshape(-1, 1)
        v_c = self.V_[c]
        return sigmoid(v_c @ vw)
    
    def loss(self, w, c, neg_samples):
        B = len(w)
        assert neg_samples.shape == (B, self.k)
        vw = self.V[:, w] # (d,B)
        v_c = self.V_[c] # (B,d)
        v_n = self.V_[neg_samples] # (B,k,d)
        p = np.sum(v_c*vw.T, axis=1) # (B,)
        n = -np.sum(v_n * vw.T[:, None, :], axis=2) # (B,k)
        pos = sigmoid(p) # (B,)
        neg = sigmoid(n) # (B,k)

        loss = (-np.log(pos).sum() - np.log(neg).sum()) / B

        return loss
    
    def train_step(self, w, c, lr):
        """
        Updates models weights according to loss for given positive samples and sampled negative samples.
        Negative samples are sampled according to unigram distribution raised to power 3/4.
        Argument:
        w - index of word; type: ndarray
        c - index of word; type: ndarray
        Returns: loss, gradients, w,c, neg_samples, details in code.
        """
        B = len(w)

        vw = self.V[:, w] # (d,B)
        v_c = self.V_[c] # (B,d)
        neg_samples = self.sample_negatives(B)
        neg_samples = neg_samples.reshape(B, self.k)
        v_n = self.V_[neg_samples] # (B,k,d)
        p = np.sum(v_c*vw.T, axis=1) # (B,)
        n = -np.sum(v_n * vw.T[:, None, :], axis=2) # (B,k)
        pos = sigmoid(p) # (B,)
        neg = sigmoid(n) # (B,k)

        loss = (-np.log(pos).sum() - np.log(neg).sum()) / B

        grad_pos = (1-pos)[:,None]       # (B,1)
        grad_neg = (1-neg)[:,:,None]  # (B,k,1)

        # gradients:
        dVw = (-(grad_pos * v_c).T + np.sum(grad_neg * v_n, axis=1).T) / B # (d,B)
        dV_c = (grad_pos * vw.T) / B # (B,d)
        dV_n = (grad_neg * vw.T[:,None,:]) / B # (B,k,d)

        # self.V[:,w] -= lr*dVw
        np.add.at(self.V, (slice(None), w), -lr * dVw)
        # self.V_[c,:] -= lr*dV_c
        np.add.at(self.V_, c, -lr * dV_c)
        # self.V_[neg_samples.reshape(-1)] -= lr * dV_n.reshape(-1,self.d) 
        np.add.at(self.V_, neg_samples.reshape(-1), -lr * dV_n.reshape(-1,self.d))
        # Repetition of words could be a problem in case of simple A += B usage, as it overwrites, not adds.
        # To prevent it I have used np.add.at(...).

        return loss, dVw, dV_c, dV_n, neg_samples
    
    def train(self, train_corpus, test_corpus, epochs, logging_freq, lr, batch_size, weight_decay=1., logdir="out/skip_gram"):
        """
        Logs information to tensorboard. Note that test loss is semi random as negative sampling is random.
        Steps are counted across all episodes. In particular some batches may consist of data from 2 episodes. 
        This is not typical approach.
        """
        writer = SummaryWriter(logdir + "/" + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        steps = 0
        centers = []
        contexts = []
        partial_steps = 0
        loss = 0.
        for e in range(epochs):
            print(f"Epoch {e}")
            for sequence_id in train_corpus:
                for i in range(len(sequence_id)):
                    w_, c_ = self.positive_samples(sequence_id, i)
                    for w,c in zip(w_,c_):
                        centers.append(w)
                        contexts.append(c)
                        partial_steps += 1
                        if partial_steps == batch_size:
                            temp_loss, _, _, _, _ = self.train_step(np.array(centers), np.array(contexts), lr)
                            loss += temp_loss
                            centers = []
                            contexts = []
                            partial_steps = 0
                            steps += 1
                            lr *= weight_decay
                        if steps % logging_freq == 0 and partial_steps == 0:
                            writer.add_scalar("train/loss", loss/logging_freq, steps)
                            print(f"loss: {loss/logging_freq}, steps: {steps}")
                            loss = 0.
            writer.add_scalar("final_step_per_episode", steps, e)
            
            test_loss = 0.
            test_steps = 0
            test_centers = []
            test_contexts = []
            test_partial_steps = 0

            for sequence_id in test_corpus:
                for i in range(len(sequence_id)):
                    w_, c_ = self.positive_samples(sequence_id, i, dynamic_window=False)
                    for w,c in zip(w_,c_):
                        test_centers.append(w)
                        test_contexts.append(c)
                        # single_updates.append(self.train_step_sgd(w,c))
                        test_partial_steps += 1
                        if test_partial_steps == batch_size:
                            neg_samples = self.sample_negatives(batch_size)
                            neg_samples = neg_samples.reshape(batch_size, self.k)
                            temp_loss = self.loss(np.array(test_centers), np.array(test_contexts), neg_samples)
                            test_loss += temp_loss
                            test_centers = []
                            test_contexts = []
                            test_partial_steps = 0
                            test_steps += 1
            if len(test_centers) > 0:
                neg_samples = self.sample_negatives(test_partial_steps)
                neg_samples = neg_samples.reshape(test_partial_steps, self.k)
                temp_loss = self.loss(np.array(test_centers), np.array(test_contexts), neg_samples)
                test_loss += temp_loss
                test_centers = []
                test_contexts = []
                test_partial_steps = 0
                test_steps += 1
            writer.add_scalar("test/loss", test_loss/test_steps, e)
            print(f"test loss: {test_loss/test_steps} (note that negative sampling is random)")
        writer.close()

