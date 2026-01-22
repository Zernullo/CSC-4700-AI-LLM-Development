"""
CSC 4700 Homework 1: N-Gram Models and BPE

Author: Daniel Guo
Instructor: Dr. Keith Mills

Section 2: Byte Pair Encoding (BPE)

This program implements Byte Pair Encoding (BPE) for subword tokenization.

Allowed libraries:
- Python standard library
- numpy
- pandas

References:
Google and ChatGPT were used to clarify Python syntax, standard library usage,
and conceptual understanding of Byte Pair Encoding (BPE).
"""

import argparse
import pickle
import re
from collections import Counter


class BPE:
    """Byte Pair Encoding (BPE) model for subword tokenization."""

    def __init__(self):
        """Initialize an empty BPE vocabulary."""
        self.vocabulary = {}

    def train(self, data_corpus, k=500):
        """
        Train the BPE model on a text corpus.

        Args:
            data_corpus (str): Input text used for training.
            k (int): Number of merge operations.
        """
        tokens = re.findall(r"\b\w+\b|[^\w\s]", data_corpus.lower())
        words_list = [list(word) + ["</w>"] for word in tokens]

        for _ in range(k):
            pairs = Counter()

            for word in words_list:
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += 1

            if not pairs:
                break

            most_frequent = pairs.most_common(1)[0][0]
            new_symbol = "".join(most_frequent)
            self.vocabulary[new_symbol] = pairs[most_frequent]

            for word in words_list:
                i = 0
                while i < len(word) - 1:
                    if (word[i], word[i + 1]) == most_frequent:
                        word[i] = word[i] + word[i + 1]
                        del word[i + 1]
                    else:
                        i += 1

    def tokenize(self, text):
        """
        Tokenize input text using the trained BPE vocabulary.

        Args:
            text (str): Input text to tokenize.

        Returns:
            tuple: List of tokens and corresponding token IDs.
        """
        words_list = [
            list(word) + ["</w>"]
            for word in re.findall(r"\b\w+\b|[^\w\s]", text.lower())
        ]

        for word in words_list:
            i = 0
            while i < len(word) - 1:
                merged = word[i] + word[i + 1]
                if merged in self.vocabulary:
                    word[i] = merged
                    del word[i + 1]
                else:
                    i += 1

        tokens = [symbol for word in words_list for symbol in word]

        token_ids = [
            list(self.vocabulary.keys()).index(tok)
            if tok in self.vocabulary
            else -1
            for tok in tokens
        ]

        return tokens, token_ids


def main():
    """Command-line interface for training and using the BPE model."""
    parser = argparse.ArgumentParser(
        description="BPE Subword Tokenization CLI"
    )

    parser.add_argument(
        "activity",
        choices=["train_bpe", "tokenize"],
        help="Select which activity to perform",
    )
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--save", type=str, help="Path to save trained model")
    parser.add_argument("--load", type=str, help="Path to load trained model")
    parser.add_argument("--text", type=str, help="Text to tokenize")
    parser.add_argument(
        "--k",
        type=int,
        default=500,
        help="Number of BPE merges (default=500)",
    )

    args = parser.parse_args()

    if args.activity == "train_bpe":
        if not args.data or not args.save:
            parser.error("train_bpe requires --data and --save")

        with open(args.data, "r", encoding="utf-8") as file:
            text = file.read()

        model = BPE()
        model.train(text, k=args.k)

        with open(args.save, "wb") as file:
            pickle.dump(model, file)

        print(f"Model trained and saved to {args.save}")

    elif args.activity == "tokenize":
        if not args.load or not args.text:
            parser.error("tokenize requires --load and --text")

        with open(args.load, "rb") as file:
            model = pickle.load(file)

        tokens, token_ids = model.tokenize(args.text)
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)


if __name__ == "__main__":
    main()
