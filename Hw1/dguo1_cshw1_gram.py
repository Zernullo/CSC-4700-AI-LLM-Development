"""
CSC 4700 Homework 1: N-Gram Models and BPE

Author: Daniel Guo
Instructor: Dr. Keith Mills

Section 1: N-Gram Models

This program implements bigram and trigram probabilistic language models.
It demonstrates how n-grams are constructed, how probabilities are calculated,
and how the model can be used to predict the next word given a context.

Allowed libraries:
- Python standard library
- numpy
- pandas

References:
Google and ChatGPT were used to clarify syntax, standard library usage,
and conceptual understanding of n-grams.
"""

import argparse
import pickle
import random
import re
from collections import defaultdict


def int_defaultdict():
    """
    Return a defaultdict(int).

    This function is defined globally because lambda functions cannot be
    pickled, and pickle requires globally defined functions.
    Using a nested lambda would fail when saving the model.
    """
    return defaultdict(int)


class NGramModel:
    """N-gram language model supporting bigrams and trigrams."""

    def __init__(self, n):
        """
        Initialize the n-gram model.

        Args:
            n (int): Order of the n-gram (2 for bigram, 3 for trigram).

        Attributes:
            vocab (set): Set of unique tokens seen during training.
            ngram_counts (dict): Nested dictionary of counts for n-grams.
            ngram_probs (dict): Nested dictionary of probabilities for n-grams.
        """
        if n not in (2, 3):
            raise ValueError("n must be 2 (bigram) or 3 (trigram)")

        self.n = n  # Store the order of the n-gram
        self.vocab = set()  # Vocabulary of unique tokens
        self.ngram_counts = defaultdict(int_defaultdict)  # Counts of observed n-grams
        self.ngram_probs = {}  # Probabilities derived from counts

    def tokenize(self, text):
        """
        Tokenize input text into words and punctuation.

        Args:
            text (str): Input text.

        Returns:
            list: List of tokens.

        Notes:
            - Converts text to lowercase for normalization.
            - Uses regex to capture words (\w+) and punctuation ([^\w\s]).
        """
        return re.findall(r"\b\w+\b|[^\w\s]", text.lower())

    def train(self, text):
        """
        Train the n-gram model on input text.

        Args:
            text (str): Training corpus.

        Steps:
            1. Tokenize text into words and punctuation.
            2. Update vocabulary with all unique tokens.
            3. Count all n-grams in the token sequence.
            4. Convert counts to probabilities.
        """
        tokens = self.tokenize(text)
        self.vocab = set(tokens)  # Keep track of unique tokens

        # Count n-grams
        for i in range(len(tokens) - self.n + 1):
            # Context is the previous n-1 words
            context = tuple(tokens[i:i + self.n - 1])
            # Next word is the current word following the context
            next_word = tokens[i + self.n - 1]
            # Increment the count for this context-next_word pair
            self.ngram_counts[context][next_word] += 1

        # Convert counts to probabilities
        self.ngram_probs = {}
        for context, next_words in self.ngram_counts.items():
            total = sum(next_words.values())
            # Normalize counts to probabilities
            self.ngram_probs[context] = {
                word: count / total
                for word, count in next_words.items()
            }

    def predict_next_word(self, context, deterministic=False):
        """
        Predict the next word given a context.

        Args:
            context (tuple): Previous word(s) forming the context.
            deterministic (bool): If True, selects the most probable next word.

        Returns:
            str: Predicted next word.

        Notes:
            - If context is not in the training data, prints an error and returns an empty string.
            - Otherwise, either samples probabilistically or chooses the max probability word.
        """
        if context not in self.ngram_probs:
            print("Error: Word not found in training data.")
            return ""

        next_words = self.ngram_probs[context]

        if deterministic:
            # Greedy selection: pick the word with highest probability
            return max(next_words, key=next_words.get)

        # Probabilistic selection: sample according to probability distribution
        words = list(next_words.keys())
        weights = list(next_words.values())
        return random.choices(words, weights=weights, k=1)[0]


def main():
    """
    Command-line interface (CLI) for training and predicting with n-gram models.

    Supports two activities:
        - train_ngram: Train a bigram or trigram model on a text corpus.
        - predict_ngram: Generate words given a starting context.
    """
    parser = argparse.ArgumentParser(
        description="N-gram Language Model CLI"
    )

    # CLI arguments
    parser.add_argument(
        "activity",
        choices=["train_ngram", "predict_ngram"],
        help="Select which activity to perform",
    )
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--save", type=str, help="Path to save trained model")
    parser.add_argument("--load", type=str, help="Path to load trained model")
    parser.add_argument("--word", type=str, help="Starting word(s) for prediction")
    parser.add_argument("--nwords", type=int, help="Number of words to predict")
    parser.add_argument("--n", type=int, choices=[2, 3], help="Order of n-gram")
    parser.add_argument("--d", action="store_true", help="Use deterministic (greedy) sampling")

    args = parser.parse_args()

    if args.activity == "train_ngram":
        # Validate required arguments for training
        if not args.data or not args.save or not args.n:
            parser.error("train_ngram requires --data, --save, and --n")

        # Read training corpus from file
        with open(args.data, "r", encoding="utf-8") as file:
            text = file.read()

        # Create and train the n-gram model
        model = NGramModel(n=args.n)
        model.train(text)

        # Save the trained model to disk
        with open(args.save, "wb") as file:
            pickle.dump(model, file)

        print(f"Model trained and saved to {args.save}")

    elif args.activity == "predict_ngram":
        # Validate required arguments for prediction
        if not args.load or not args.word or not args.nwords:
            parser.error("predict_ngram requires --load, --word, and --nwords")

        # Load the trained model from disk
        with open(args.load, "rb") as file:
            model = pickle.load(file)

        # Split the starting word(s) into context tuple
        context = tuple(args.word.lower().split())
        generated = list(context)  # Initialize generated sequence with starting context

        # Generate words iteratively
        for _ in range(args.nwords):
            # Take the last n-1 words as context
            next_word = model.predict_next_word(
                tuple(generated[-(model.n - 1):]),
                deterministic=args.d,
            )
            if not next_word:
                break  # Stop if model cannot predict next word
            generated.append(next_word)

        # Output generated sequence as a single string
        print(" ".join(generated))


if __name__ == "__main__":
    main()
