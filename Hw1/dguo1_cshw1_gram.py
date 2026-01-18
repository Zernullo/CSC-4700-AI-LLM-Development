"""
CSC 4700 Homework 1: n-gram Language Models
Author: Daniel Guo
Instructor: Dr. Keith Mills

Implement bigram and trigram probabilistic language models and understand how they 
operate on a code-level.

Allowed libraries: Python standard library, numpy, pandas

References: Google, ChatGPT
Google and ChatGPT were used to clarify syntax and usage of standard library functions.
As well as to understand concepts of n-grams
"""


import argparse
import pickle
import random
import re
from collections import defaultdict

class NGramModel:
    def __init__(self, n):
        if n not in (2, 3):
            raise ValueError("n must be 2 (bigram) or 3 (trigram)")
        self.n = n # n-gram size
        self.vocab = set() # Vocabulary set, help to track words seen during training so bascially no duplicates words
        self.ngram_counts = defaultdict(lambda: defaultdict(int)) # Nested dictionary for n-gram counts
        self.ngram_probs = {} # Nested dictionary for n-gram probabilities

    # Tokenizes the input text into tokens (words and punctuation).
    def tokenize(self, text): 
        return re.findall(r"\b\w+\b|[^\w\s]", text.lower()) # Lowercase for normalization, \b\w+\b|[^\w\s] - \b\w+\b matches words, [^\w\s] matches punctuation

    # Trains the n-gram model on the provided text.
    def train(self, text):
        # Counts n-gram occurrences in the list of tokens.
        tokens = self.tokenize(text)
        self.vocab = set(tokens)  # Update vocabulary with unique tokens
        for i in range(len(tokens) - self.n + 1): # Loop through tokens to get n-grams
            current_word = tuple(tokens[i:i+self.n-1]) 
            next_word = tokens[i + self.n - 1] 
            self.ngram_counts[current_word][next_word] += 1

        # Converting counts to probabilities.
        """
        Converts a nested dictionary of counts into probabilities.
        Input: {current_word: {next_word: count}}
        Output: {current_word: {next_word: probability}}

        Dictionary comprehension:
        probabilities[current_word] = {
            word: count / total
            for word, count in next_words.items()
        }

        Equivalent normal loop:
        inner_dict = {}
        for word, count in next_words.items():
            inner_dict[word] = count / total
        probabilities[current_word] = inner_dict
        """
        self.ngram_probs = {}
        for current_word, next_words in self.ngram_counts.items():
            total = sum(next_words.values())
            self.ngram_probs[current_word] = {
                word: count / total
                for word, count in next_words.items()
            }

    # Predicts the next word based on the current word and probability dictionary.
    def predict_next_word(self, input, deterministic=False):

        if input not in self.ngram_probs:
            print("Error: Word not found in training data.")
            return ""
        
        next_words = self.ngram_probs[input]
        
        if deterministic:
            # Greedy argmax
            return max(next_words, key=next_words.get) # Get the word with the highest probability
        
        # Categorical sampling
        words = list(next_words.keys())
        weights = list(next_words.values())
        return random.choices(words, weights=weights, k=1)[0]


# Command-line interface for training and predicting with n-gram models.
def main():
    parser = argparse.ArgumentParser(description="N-gram Language Model CLI") # argparse is Python standard library for parsing command-line arguments
    
    # add_argument method defines what arguments the program requires
    parser.add_argument(
        "activity",
        choices=["train_ngram", "predict_ngram"],
        help="Select which activity to perform"
    )
    parser.add_argument("--data", type=str, help="Path to training data") # Path to training data
    parser.add_argument("--save", type=str, help="Path to save trained model") # Path to save trained model
    parser.add_argument("--load", type=str, help="Path to load trained model") # Path to load trained model
    parser.add_argument("--word", type=str, help="Starting word(s) for prediction") # Starting word(s) for prediction ex. --word whale (bigram) or --word whale of (trigram)
    parser.add_argument("--nwords", type=int, help="Number of words to predict") # Number of words to generate during prediction ex. --nwords 10 (generate 10 words)
    parser.add_argument("--n", type=int, choices=[2, 3], help="Order of n-gram") # choices limits input to 2 or 3
    parser.add_argument("--d", action="store_true", help="Use deterministic (greedy) sampling") # action="store_true" means if user includes --d in command line, args.d will be True, otherwise False

    args = parser.parse_args()

    if args.activity == "train_ngram":
        if not args.data or not args.save or not args.n:
            parser.error("train_ngram requires --data, --save, and --n")
        
        # Read file in order to train model
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Train model
        model = NGramModel(n=args.n) # Create n-gram model instance base on n value choosen in the parser.add_argument("--n", ...)
        model.train(text)
        
        # Save model after training
        with open(args.save, "wb") as f: # "wb" means write binary mode
            pickle.dump(model, f) # pickle.dump to serialize the model object to file, meaning save the model to disk
        
        print(f"Model trained and saved to {args.save}")

    elif args.activity == "predict_ngram":
        if not args.load or not args.word or not args.nwords:
            parser.error("predict_ngram requires --load, --word, and --nwords")
        
        # Load saved trained model
        with open(args.load, "rb") as f: # "rb" means read binary mode
            model = pickle.load(f) # pickle.load to deserialize the model object from file, meaning load the model back into memory
        
        # Split starting words
        context = tuple(args.word.lower().split()) # Lowercase for normalization, split into tuple of words
        generated = list(context)
        
        for _ in range(args.nwords): # Generate nwords number of words
            # Predict next word using the model, chosen deterministic in the parser.add_argument("--d", ...)
            next_word = model.predict_next_word(tuple(generated[-(model.n-1):]), deterministic=args.d) # gernerated [-(model.n-1):] gets the last n-1 words from generated list
            if not next_word:
                break
            generated.append(next_word)
        
        print(" ".join(generated)) # Join the generated words into a single string and print



if __name__ == "__main__":
    main()
