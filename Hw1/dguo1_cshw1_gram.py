"""
CSC 4700 Homework 1: N-Gram Models and BPE

Author: Daniel Guo
Instructor: Dr. Keith Mills

Section 1: N-Gram Models

This program implements bigram and trigram probabilistic language models
and demonstrates how they operate at the code level.

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
    """
    return defaultdict(int)


class NGramModel:
    """N-gram language model supporting bigrams and trigrams."""

    def __init__(self, n):
        """
        Initialize the n-gram model.

        Args:
            n (int): Order of the n-gram (2 for bigram, 3 for trigram).
        """
        if n not in (2, 3):
            raise ValueError("n must be 2 (bigram) or 3 (trigram)")

        self.n = n
        self.vocab = set()
        self.ngram_counts = defaultdict(int_defaultdict)
        self.ngram_probs = {}

    def tokenize(self, text):
        """
        Tokenize input text into words and punctuation.

        Args:
            text (str): Input text.

        Returns:
            list: List of tokens.
        """
        return re.findall(r"\b\w+\b|[^\w\s]", text.lower())

    def train(self, text):
        """
        Train the n-gram model on input text.

        Args:
            text (str): Training corpus.
        """
        tokens = self.tokenize(text)
        self.vocab = set(tokens)

        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            next_word = tokens[i + self.n - 1]
            self.ngram_counts[context][next_word] += 1

        self.ngram_probs = {}
        for context, next_words in self.ngram_counts.items():
            total = sum(next_words.values())
            self.ngram_probs[context] = {
                word: count / total
                for word, count in next_words.items()
            }

    def predict_next_word(self, context, deterministic=False):
        """
        Predict the next word given a context.

        Args:
            context (tuple): Previous word(s).
            deterministic (bool): If True, use greedy selection.

        Returns:
            str: Predicted next word.
        """
        if context not in self.ngram_probs:
            print("Error: Word not found in training data.")
            return ""

        next_words = self.ngram_probs[context]

        if deterministic:
            return max(next_words, key=next_words.get)

        words = list(next_words.keys())
        weights = list(next_words.values())
        return random.choices(words, weights=weights, k=1)[0]


def main():
    """Command-line interface for training and predicting with n-gram models."""
    parser = argparse.ArgumentParser(
        description="N-gram Language Model CLI"
    )

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
        if not args.data or not args.save or not args.n:
            parser.error("train_ngram requires --data, --save, and --n")

        with open(args.data, "r", encoding="utf-8") as file:
            text = file.read()

        model = NGramModel(n=args.n)
        model.train(text)

        with open(args.save, "wb") as file:
            pickle.dump(model, file)

        print(f"Model trained and saved to {args.save}")

    elif args.activity == "predict_ngram":
        if not args.load or not args.word or not args.nwords:
            parser.error("predict_ngram requires --load, --word, and --nwords")

        with open(args.load, "rb") as file:
            model = pickle.load(file)

        context = tuple(args.word.lower().split())
        generated = list(context)

        for _ in range(args.nwords):
            next_word = model.predict_next_word(
                tuple(generated[-(model.n - 1):]),
                deterministic=args.d,
            )
            if not next_word:
                break
            generated.append(next_word)

        print(" ".join(generated))


if __name__ == "__main__":
    main()
