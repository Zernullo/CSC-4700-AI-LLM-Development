"""
CSC 4700 Homework 1: N-Gram Models and BPE
Author: Daniel Guo
Instructor: Dr. Keith Mills

Section 2: Byte Pair Encoding (BPE)
Implement Byte Pair Encoding (BPE) for subword tokenization and understand its application

Allowed libraries: Python standard library, numpy, pandas

References: Google, ChatGPT
Google and ChatGPT were used to clarify syntax and usage of standard library functions.
As well as to understand concepts of Byte Pair Encoding (BPE)
"""

import argparse
import pickle
import re
from collections import defaultdict, Counter

class BPE:
    def __init__ (self):
        self.vocabulary = {}  # Vocabulary with word frequencies
    
    # Trains the BPE model on the provided data corpus for k merges.
    def train(self, data_corpus, k=500):
        words = re.findall(r"\b\w+\b|[^\w\s]", data_corpus.lower()) # Tokenize the corpus into words and punctuation
        words_list = [list(word) + ['</w>'] for word in words] # split each word into list of characters with end-of-word token

        for _ in range(k):
            pairs = Counter()
            # index in words_list
            for word in words_list:
                # character index in word
                for i in range(len(word)-1):
                    pairs[(word[i], word[i+1])] += 1 # Count frequency of each pair of symbols
            
            if not pairs: break  # No more pairs to merge
            
            """
            Gets the most frequent pair of symbols.
            Creates a new symbol by merging the most frequent pair.
            Updates the vocabulary with the new symbol and its frequency.

            self.vocabulary contains all merged symbols that your BPE model has discovered during training.
            Each key is a token, each value is how often that token appeared.
            You need to know: what tokens exist so you can split the text into the biggest matching pieces.
            Without self.vocabulary, the BPE tokenizer wouldn't know that 'th', 'he', or 'wh' are valid tokens 
            — it would just split everything into single characters again.
            """
            most_frequent = pairs.most_common(1)[0][0]
            new_symbol = ''.join(most_frequent) 
            self.vocabulary[new_symbol] = pairs[most_frequent]
            
            # Update words_list by merging the most frequent pair. So that next iteration of BPE can see the new merged symbols insstead of just single characters
            for word in words_list:
                i = 0
                while i < len(word) - 1:
                    if (word[i], word[i+1]) == most_frequent:
                        # Merge symbols
                        word[i] = word[i] + word[i+1]
                        del word[i+1]
                    else:
                        i += 1

    # Tokenizes the input text into BPE subword tokens.
    def tokenize(self, text):
        # words_list is a list of lists, where each inner list contains the characters of a word plus the end-of-word token '</w>'
        words_list = [list(word) + ['</w>'] for word in re.findall(r"\b\w+\b|[^\w\s]", text.lower())]

        # Merge symbols in words_list based on the learned vocabulary
        for word in words_list:
            i = 0
            while i < len(word) - 1:
                pair = (word[i], word[i+1])
                merged = ''.join(pair)
                # Check if the merged symbol is in the vocabulary
                if merged in self.vocabulary:
                    word[i] = merged
                    del word[i+1]
                else:
                    i += 1
        
        """
        Flattens the list of lists into a single list of tokens. Means converting words_list (list of lists) into tokens (single list).
        Example: 
        words_list = [['h', 'e', 'l', 'l', 'o', '</w>'], ['w', 'o', 'r', 'l', 'd', '</w>']]
        Resulting tokens = ['h', 'e', 'l', 'l', 'o', '</w>', 'w', 'o', 'r', 'l', 'd', '</w>']

        for word in words_list -> loops through each word (which is a list of symbols)
        for symbol in word -> loops through each symbol in the word
        symbol -> adds the chars to the tokens list

        for symbol in word:
            for word in words_list:
                tokens.append(symbol)
        """
        tokens = [symbol for word in words_list for symbol in word]
        
        """
        Converts tokens to their corresponding IDs based on the vocabulary.
        for tok in tokens -> loops through each token
        if tok in self.vocabulary -> checks if the token exists in the vocabulary
        list(self.vocabulary.keys()).index(tok) -> gets the index of the token in the vocabulary keys
        else -1 -> if token not found in vocabulary, assign -1 as its ID

        for tok in tokens:
            if tok in self.vocabulary:
                token_ids.append(list(self.vocabulary.keys()).index(tok))
            else:
                token_ids.append(-1)
        """
        token_ids = [list(self.vocabulary.keys()).index(tok) if tok in self.vocabulary else -1 for tok in tokens]
        return tokens, token_ids
    
# Command-line interface for training and predicting with n-gram models.
def main():
    parser = argparse.ArgumentParser(description="BPE Subword Tokenization CLI") # argparse is Python standard library for parsing command-line arguments
    
    # add_argument method defines what arguments the program requires
    parser.add_argument(
        "activity",
        choices=["train_bpe", "tokenize"],
        help="Select which activity to perform"
    )
    parser.add_argument("--data", type=str, help="Path to training data") # Path to training data
    parser.add_argument("--save", type=str, help="Path to save trained model") # Path to save trained model
    parser.add_argument("--load", type=str, help="Path to load trained model") # Path to load trained model
    parser.add_argument("--text", type=str, help="Text to tokenize") # Text to tokenize
    parser.add_argument("--k", type=int, default=500, help="Number of BPE merges (default=500)")

    args = parser.parse_args()

    if args.activity == "train_bpe":
        if not args.data or not args.save:
            parser.error("train_bpe requires --data and --save")
        
        # Read file in order to train model
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Train model
        model = BPE() # Create BPE model instance
        model.train(text, k=args.k) # Train BPE model with k merges
        
        # Save model after training
        with open(args.save, "wb") as f: # "wb" means write binary mode
            pickle.dump(model, f) # pickle.dump to serialize the model object to file, meaning save the model to disk
        
        print(f"Model trained and saved to {args.save}")

    elif args.activity == "tokenize":
        if not args.load or not args.text:
            parser.error("tokenize requires --load and --text")
        
        # Load saved trained model
        with open(args.load, "rb") as f: # "rb" means read binary mode
            model = pickle.load(f) # pickle.load to deserialize the model object from file, meaning load the model back into memory
        
        tokens, token_ids = model.tokenize(args.text) #
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)

if __name__ == "__main__":
    main()
