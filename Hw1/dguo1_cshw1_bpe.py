"""
CSC 4700 Homework 1: N-Gram Models and BPE

Author: Daniel Guo
Instructor: Dr. Keith Mills

Section 2: Byte Pair Encoding (BPE)

This program implements Byte Pair Encoding (BPE) for subword tokenization.
It allows training a BPE model on a text corpus and then using the model
to tokenize new text into subword units, with corresponding token IDs.

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
        """Initialize an empty BPE vocabulary to store learned subword merges."""
        self.vocabulary = {}  # Dictionary: key = merged symbol, value = frequency count

    def train(self, data_corpus, k=500):
        """
        Train the BPE model on a text corpus.

        Args:
            data_corpus (str): Input text used for training.
            k (int): Number of merge operations to perform.

        The method repeatedly finds the most frequent pair of symbols in the
        corpus, merges them into a new symbol, and updates the vocabulary.
        """
        # Tokenize text into words and punctuation symbols
        tokens = re.findall(r"\b\w+\b|[^\w\s]", data_corpus.lower())
        # Split each word into a list of characters and append the end-of-word marker
        words_list = [list(word) + ["</w>"] for word in tokens]

        for _ in range(k):
            pairs = Counter()  # Count occurrences of adjacent symbol pairs

            # Count all pairs in the current words_list
            for word in words_list:
                for i in range(len(word) - 1):
                    # Skip pairs containing the end-of-word marker to prevent invalid merges
                    if word[i] == "</w>" or word[i + 1] == "</w>":
                        continue
                    pairs[(word[i], word[i + 1])] += 1

            # Stop if no more pairs to merge
            if not pairs:
                break

            # Identify the most frequent pair and merge it
            most_frequent = pairs.most_common(1)[0][0]
            new_symbol = "".join(most_frequent)
            self.vocabulary[new_symbol] = pairs[most_frequent]

            # Apply the merge to the words_list so that the new symbol can participate in future merges
            for word in words_list:
                i = 0
                while i < len(word) - 1:
                    if (word[i], word[i + 1]) == most_frequent:
                        # Replace the pair with the merged symbol
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
            tuple: (tokens, token_ids)
                tokens (list): List of subword tokens.
                token_ids (list): Corresponding IDs of tokens in the vocabulary.
        
        The method splits words into characters, applies BPE merges learned
        during training, removes end-of-word markers, and returns token IDs.
        """
        # Split text into words and punctuation symbols with end-of-word marker
        words_list = [
            list(word) + ["</w>"]
            for word in re.findall(r"\b\w+\b|[^\w\s]", text.lower())
        ]

        # Merge symbols according to the trained BPE vocabulary
        for word in words_list:
            i = 0
            while i < len(word) - 1:
                # Skip end-of-word marker to prevent invalid merges
                if word[i] == "</w>" or word[i + 1] == "</w>":
                    i += 1
                    continue
                merged = word[i] + word[i + 1]
                if merged in self.vocabulary:
                    word[i] = merged
                    del word[i + 1]
                else:
                    i += 1

        # Flatten list of lists into a single list of tokens
        tokens = [symbol for word in words_list for symbol in word]
        # Remove end-of-word markers for clean output
        tokens = [t for t in tokens if t != '</w>']

        # Convert tokens to IDs based on the order in the vocabulary
        token_ids = [
            list(self.vocabulary.keys()).index(tok)
            if tok in self.vocabulary
            else -1
            for tok in tokens
        ]
        
        return tokens, token_ids


def main():
    """Command-line interface (CLI) for training and using the BPE model."""
    parser = argparse.ArgumentParser(
        description="BPE Subword Tokenization CLI"
    )

    # Define CLI arguments
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
        # Ensure both data and save path are provided
        if not args.data or not args.save:
            parser.error("train_bpe requires --data and --save")

        # Read the input corpus from file
        with open(args.data, "r", encoding="utf-8") as file:
            text = file.read()

        # Train a new BPE model
        model = BPE()
        model.train(text, k=args.k)

        # Save the trained model to disk
        with open(args.save, "wb") as file:
            pickle.dump(model, file)

        print(f"Model trained and saved to {args.save}")

    elif args.activity == "tokenize":
        # Ensure model path and text are provided
        if not args.load or not args.text:
            parser.error("tokenize requires --load and --text")

        # Load the trained BPE model
        with open(args.load, "rb") as file:
            model = pickle.load(file)

        # Tokenize the input text
        tokens, token_ids = model.tokenize(args.text)
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)


if __name__ == "__main__":
    main()
