import numpy as np
import random
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, max_length=200):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            word_freq.update(words)

        # Get most common words
        most_common = word_freq.most_common(self.max_vocab_size - 2)

        for idx, (word, freq) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)}")

    def encode(self, texts):
        """Convert texts to sequences of indices"""
        sequences = []
        for text in texts:
            words = text.lower().split()
            seq = [self.word2idx.get(word, 1) for word in words]

            # Pad or truncate
            if len(seq) < self.max_length:
                seq = seq + [0] * (self.max_length - len(seq))
            else:
                seq = seq[:self.max_length]

            sequences.append(seq)

        return np.array(sequences)

def load_data(train_size=20000, test_size=0.2, random_state=42):
    """
    Load dataset, preprocess, and split into train/val.
    Returns X_train, y_train, X_val, y_val, vocab_size, preprocessor
    """
    # Load and explore the dataset
    print("Loading dataset...")
    dataset = load_dataset('sh0416/ag_news')

    # Sample data
    # Ensure random state for reproducibility of sampling
    random.seed(random_state)
    
    # Check if we have enough data, otherwise take all
    available_size = len(dataset['train'])
    if train_size > available_size:
        print(f"Requested train_size {train_size} > available {available_size}. Using all data.")
        train_indices = range(available_size)
    else:
        train_indices = random.sample(range(available_size), train_size)
    
    train_data = dataset['train'].select(train_indices)

    # Prepare texts and labels
    train_texts = [item['title'] + ' ' + item['description'] for item in train_data]
    train_labels = [item['label'] - 1 for item in train_data]  # Convert to 0-indexed

    # Split into train and validation
    print("Splitting data...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=test_size, random_state=random_state, stratify=train_labels
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Build vocabulary and encode texts
    print("Building vocabulary and encoding...")
    preprocessor = TextPreprocessor(max_vocab_size=10000, max_length=200)
    preprocessor.build_vocab(train_texts)

    X_train = preprocessor.encode(train_texts)
    X_val = preprocessor.encode(val_texts)
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    vocab_size = len(preprocessor.word2idx)

    return X_train, y_train, X_val, y_val, vocab_size, preprocessor
