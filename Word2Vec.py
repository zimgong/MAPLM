import numpy as np
import torch

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from collections import Counter
import random

# Sort of smart tokenization
from nltk.tokenize import RegexpTokenizer

#
# IMPORTANT NOTE: Always set your random seeds when dealing with stochastic
# algorithms as it lets your bugs be reproducible and (more importantly) it lets
# your results be reproducible by others.
#
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)


class RandomNumberGenerator:
    ''' 
    A wrapper class for a random number generator that will (eventually) hold buffers of pre-generated random numbers for
    faster access. For now, it just calls np.random.randint and np.random.random to generate these numbers 
    at the time they are needed.
    '''

    def __init__(self, buffer_size, seed=12345):
        '''
        Initializes the random number generator with a seed and a buffer size of random numbers to use

        Args:
            buffer_size: The number of random numbers to pre-generate. You will eventually want 
                         this to be a large-enough number than you're not frequently regenerating the buffer
            seed: The seed for the random number generator
        '''
        self.max_val = -1
        # TODO (later): create a random number generator using numpy and set its seed
        # TODO (later): pre-generate a buffer of random floats to use for random()
        self.rng = np.random.default_rng(seed)
        self.buffer_size = buffer_size
        self.buffer_index = 0

    def random(self):
        '''
        Returns a random float value between 0 and 1
        '''
        # TODO (later): get a random number from the float buffer, rather than calling np.random.random
        # NOTE: If you reach the end of the buffer, you should refill it with new random float numbers
        if self.buffer_index == len(self.int_buffer):
            self.int_buffer = self.rng.integers(
                0, self.max_val, self.buffer_size)
            self.buffer_index = 0
        val = self.int_buffer[self.buffer_index] / self.max_val
        self.buffer_index += 1
        return val

    def set_max_val(self, max_val):
        '''
        Sets the maximum integer value for randint and creates a buffer of random integers
        '''
        self.max_val = max_val
        # NOTE: This default implemenation just sets the max_val and does not create a buffer of random integers
        # TODO (later): Implement a buffer of random integers (for now, we'll just use np.random.randint)
        self.int_buffer = self.rng.integers(0, max_val, self.buffer_size)

    def randint(self):
        '''
        Returns a random int value between 0 and self.max_val (inclusive)
        '''
        if self.max_val == -1:
            raise ValueError("Need to call set_max_val before calling randint")

        # TODO (later): get a random number from the int buffer, rather than calling np.random.randint
        # NOTE: If you reach the end of the buffer, you should refill it with new random ints
        if self.buffer_index == len(self.int_buffer):
            self.int_buffer = self.rng.integers(
                0, self.max_val, self.buffer_size)
            self.buffer_index = 0
        val = self.int_buffer[self.buffer_index]
        self.buffer_index += 1
        return val


class Corpus:

    def __init__(self, rng: RandomNumberGenerator):

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.rng = rng

        # These state variables become populated with function calls
        #
        # 1. load_data()
        # 2. generate_negative_sampling_table()
        #
        # See those functions for how the various values get filled in

        self.word_to_index = {}  # word to unique-id
        self.index_to_word = {}  # unique-id to word

        # How many times each word occurs in our data after filtering
        self.word_counts = Counter()

        # A utility data structure that lets us quickly sample "negative"
        # instances in a context. This table contains unique-ids
        self.negative_sampling_table = []

        # The dataset we'll use for training, as a sequence of unqiue word
        # ids. This is the sequence across all documents after tokens have been
        # randomly subsampled by the word2vec preprocessing step
        self.full_token_sequence_as_ids = None

    def tokenize(self, text):
        '''
        Tokenize the document and returns a list of the tokens
        '''
        return self.tokenizer.tokenize(text)

    def load_data(self, file_name, min_token_freq):
        '''
        Reads the data from the specified file as long long sequence of text
        (ignoring line breaks) and populates the data structures of this
        word2vec object.
        '''

        # Step 1: Read in the file and create a long sequence of tokens for
        # all tokens in the file
        all_tokens = []
        print('Reading data and tokenizing')
        with open(file_name, 'r') as f:
            for line in f:
                all_tokens.extend(self.tokenize(line))

        # Step 2: Count how many tokens we have of each type
        print('Counting token frequencies')
        self.word_counts = Counter(all_tokens)

        # Step 3: Replace all tokens below the specified frequency with an <UNK>
        # token.
        #
        # NOTE: You can do this step later if needed
        print("Performing minimum thresholding")
        all_tokens = [token if self.word_counts[token] >=
                      min_token_freq else "<UNK>" for token in all_tokens]

        # Step 4: update self.word_counts to be the number of times each word
        # occurs (including <UNK>)
        self.word_counts = Counter(all_tokens)

        # Step 5: Create the mappings from word to unique integer ID and the
        # reverse mapping.
        for idx, token in tqdm(enumerate(self.word_counts.keys())):
            self.word_to_index[token] = idx
            self.index_to_word[idx] = token

        # Step 6: Compute the probability of keeping any particular *token* of a
        # word in the training sequence, which we'll use to subsample. This subsampling
        # avoids having the training data be filled with many overly common words
        # as positive examples in the context
        sub_sampling_prob = {}
        for token in self.word_to_index:
            freq = self.word_counts[token] / len(all_tokens)
            sub_sampling_prob[self.word_to_index[token]] = (
                np.sqrt(freq / 0.001) + 1) * 0.001 / freq

        # Step 7: process the list of tokens (after min-freq filtering) to fill
        # a new list self.full_token_sequence_as_ids where
        #
        # (1) we probabilistically choose whether to keep each *token* based on the
        # subsampling probabilities (note that this does not mean we drop
        # an entire word!) and
        #
        # (2) all tokens are convered to their unique ids for faster training.
        #
        # NOTE: You can skip the subsampling part and just do step 2 to get
        # your model up and running.
        self.full_token_sequence_as_ids = []
        for token in tqdm(all_tokens):
            if self.rng.random() < sub_sampling_prob[self.word_to_index[token]]:
                self.full_token_sequence_as_ids.append(
                    self.word_to_index[token])

        # NOTE 2: You will perform token-based subsampling based on the probabilities in
        # word_to_sample_prob. When subsampling, you are modifying the sequence itself
        # (like deleting an item in a list). This action effectively makes the context
        # window  larger for some target words by removing context words that are common
        # from a particular context before the training occurs (which then would now include
        # other words that were previously just outside the window).

        # Helpful print statement to verify what you've loaded
        print('Loaded all data from %s; saw %d tokens (%d unique)'
              % (file_name, len(self.full_token_sequence_as_ids),
                 len(self.word_to_index)))

    def generate_negative_sampling_table(self, exp_power=0.75, table_size=1e6):
        '''
        Generates a big list data structure that we can quickly randomly index into
        in order to select a negative training example (i.e., a word that was
        *not* present in the context). 
        '''

        # Step 1: Figure out how many instances of each word need to go into the
        # negative sampling table.
        #
        # HINT: np.power and np.fill might be useful here
        print("Generating sampling table")
        tokens_count = np.array(list(self.word_counts.keys()))
        inst_count = np.array(list(self.word_counts.values()))
        inst_count = np.power(inst_count, exp_power) / np.sum(inst_count)
        inst_count = np.floor(inst_count * table_size).astype(np.int32)
        tokens_count = np.array([self.word_to_index[token]
                                for token in tokens_count])

        # Step 2: Create the table to the correct size. You'll want this to be a
        # numpy array of type int
        pdf_counter = Counter(dict(zip(tokens_count, inst_count)))

        # Step 3: Fill the table so that each word has a number of IDs
        # proportionate to its probability of being sampled.
        #
        # Example: if we have 3 words "a" "b" and "c" with probabilites 0.5,
        # 0.33, 0.16 and a table size of 6 then our table would look like this
        # (before converting the words to IDs):
        #
        # [ "a", "a", "a", "b", "b", "c" ]
        self.negative_sampling_table = np.array(list(pdf_counter.elements()))
        self.rng.set_max_val(len(self.negative_sampling_table))

    def generate_negative_samples(self, cur_context_word_id, num_samples):
        '''
        Randomly samples the specified number of negative samples from the lookup
        table and returns this list of IDs as a numpy array. As a performance
        improvement, avoid sampling a negative example that has the same ID as
        the current positive context word.
        '''

        results = []

        # Create a list and sample from the negative_sampling_table to
        # grow the list to num_samples, avoiding adding a negative example that
        # has the same ID as the current context_word
        while len(results) < num_samples:
            sample = self.rng.randint()
            if sample != cur_context_word_id:
                results.append(self.negative_sampling_table[sample])

        return results


class Word2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super(Word2Vec, self).__init__()

        # Save what state you want and create the embeddings for your
        # target and context words
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.target_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_size)

        # Once created, let's fill the embeddings with non-zero random
        # numbers. We need to do this to get the training started.
        #
        # NOTE: Why do this? Think about what happens if all the embeddings
        # are all zeros initially. What would the predictions look like for
        # word2vec with these embeddings and how would the updated work?

        self.init_emb(init_range=0.5/self.vocab_size)

    def init_emb(self, init_range):

        # Fill your two embeddings with random numbers uniformly sampled
        # between +/- init_range

        self.target_embeddings.weight = nn.parameter.Parameter(torch.rand(
            self.vocab_size, self.embedding_size) * 2 * init_range - init_range)
        self.context_embeddings.weight = nn.parameter.Parameter(torch.rand(
            self.vocab_size, self.embedding_size) * 2 * init_range - init_range)

    def forward(self, target_word_id, context_word_ids):
        ''' 
        Predicts whether each context word was actually in the context of the target word.
        The input is a tensor with a single target word's id and a tensor containing each
        of the context words' ids (this includes both positive and negative examples).
        '''

        # NOTE 1: This is probably the hardest part of the homework, so you'll
        # need to figure out how to do the dot-product between embeddings and return
        # the sigmoid. Be prepared for lots of debugging. For some reference,
        # our implementation is three lines and really the hard part is just
        # the last line. However, it's usually a matter of figuring out what
        # that one line looks like that ends up being the hard part.

        # NOTE 2: In this homework you'll be dealing with *batches* of instances
        # rather than a single instance at once. PyTorch mostly handles this
        # seamlessly under the hood for you (which is very nice) but batching
        # can show in weird ways and create challenges in debugging initially.
        # For one, your inputs will get an extra dimension. So, for example,
        # if you have a batch size of 4, your input for target_word_id will
        # really be 4 x 1. If you get the embeddings of those targets,
        # it then becomes 4x50! The same applies to the context_word_ids, except
        # that was alreayd a list so now you have things with shape
        #
        #    (batch x context_words x embedding_size)
        #
        # One of your tasks will be to figure out how to get things lined up
        # so everything "just works". When it does, the code looks surprisingly
        # simple, but it might take a lot of debugging (or not!) to get there.

        # NOTE 3: We *strongly* discourage you from looking for existing
        # implementations of word2vec online. Sadly, having reviewed most of the
        # highly-visible ones, they are actually wrong (wow!) or are doing
        # inefficient things like computing the full softmax instead of doing
        # the negative sampling. Looking at these will likely leave you more
        # confused than if you just tried to figure it out yourself.

        # NOTE 4: There many ways to implement this, some more efficient
        # than others. You will want to get it working first and then
        # test the timing to see how long it takes. As long as the
        # code works (vector comparisons look good) you'll receive full
        # credit. However, very slow implementations may take hours(!)
        # to converge so plan ahead.

        # Hint 1: You may want to review the mathematical operations on how
        # to compute the dot product to see how to do these

        # Hint 2: the "dim" argument for some operations may come in handy,
        # depending on your implementation

        # TODO: Implement the forward pass of word2vec
        h = self.target_embeddings(target_word_id)
        u = self.context_embeddings(context_word_ids)

        dot_product = torch.bmm(h.unsqueeze(1), u.transpose(1, 2)).squeeze(1)

        predictions = torch.sigmoid(dot_product)

        return predictions
