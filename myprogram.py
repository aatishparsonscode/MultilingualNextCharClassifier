#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import re
import torch
import torch.nn.functional as F
import pickle
import traceback
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter, defaultdict
import json

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
except Exception as e:
    print(f"Error detecting device, defaulting to CPU: {e}")
    device = torch.device("cpu")
# Detect if CUDA is available and set device accordingly
try:
    if torch.cuda.is_available():
        # Setup scaler for stable float16 training
        from torch.cuda.amp import autocast, GradScaler
        print("Mixed precision training (float16) enabled")
        use_amp = True
    else:
        print("Mixed precision not available on CPU, using full precision")
        use_amp = False
except Exception as e:
    print(f"Error setting up mixed precision: {e}")
    use_amp = False
    device = torch.device("cpu")

class WordLevelCharPredictor(nn.Module):
    """LSTM model using word-level inputs but predicting the next character."""
    def __init__(self, word_vocab_size, char_vocab_size, embedding_dim, hidden_dim, sequence_length):
        super(WordLevelCharPredictor, self).__init__()
        # Add validation for inputs
        if word_vocab_size <= 0 or char_vocab_size <= 0:
            raise ValueError("Vocabulary sizes must be positive integers")
        if embedding_dim <= 0 or hidden_dim <= 0 or sequence_length <= 0:
            raise ValueError("Dimensions must be positive integers")
            
        self.embedding = nn.Embedding(word_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # The final layer predicts characters, not words
        self.fc = nn.Linear(hidden_dim, char_vocab_size)
    
    def forward(self, x):
        try:
            # Input validation
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")
                
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # Get the last output in the sequence
            x = self.fc(x)
            return x
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            # Return zeros as a fallback
            return torch.zeros(x.size(0), self.fc.out_features, device=x.device)

class WordTokenizer:
    def __init__(self, oov_token="<UNK>", pad_token="<PAD>", max_vocab_size=25000):
        self.word2idx = {}
        self.idx2word = {}
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.max_vocab_size = max_vocab_size  # Limit vocabulary size
        self.word_counts = Counter()  # Track word frequencies
        
    def word_in_vocab(self, word):
        return word in self.word2idx

    def fit_on_texts(self, texts):
        if not texts:
            raise ValueError("Cannot fit tokenizer on empty text list")
        
        # Count word frequencies first
        for text in texts:
            if not isinstance(text, str):
                warnings.warn(f"Non-string input detected: {type(text)}. Converting to string.")
                text = str(text)
            words = text.split()
            self.word_counts.update(words)
        
        # Initialize with special tokens
        self.word2idx = {self.pad_token: 0, self.oov_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.oov_token}
        
        # Get most common words up to max_vocab_size - 2 (accounting for special tokens)
        most_common_words = [word for word, _ in self.word_counts.most_common(self.max_vocab_size - 2)]
        
        # Build vocabulary with most common words
        for idx, word in enumerate(most_common_words, 2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary limited to {len(self.word2idx)} words (from {len(self.word_counts)} unique words)")

    def texts_to_sequences(self, texts, maxlen=None):
        sequences = []
        for text in texts:
            if not isinstance(text, str):
                warnings.warn(f"Non-string input detected: {type(text)}. Converting to string.")
                text = str(text)
                
            words = text.split()
            sequence = [self.word2idx.get(word, self.word2idx[self.oov_token]) for word in words]
            if maxlen:
                sequence = sequence[-maxlen:]
                sequence = [self.word2idx[self.pad_token]] * (maxlen - len(sequence)) + sequence
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        return [' '.join([self.idx2word.get(idx, self.idx2word.get(1, "<UNK>")) for idx in seq]) for seq in sequences]
    
    def vocab_size(self):
        return len(self.word2idx)

class CharTokenizer:
    def __init__(self, oov_token="<UNK>", pad_token="<PAD>"):
        self.char2idx = {}
        self.idx2char = {}
        self.oov_token = oov_token
        self.pad_token = pad_token
        
    def char_in_vocab(self, char):
        return char in self.char2idx

    def fit_on_texts(self, texts):
        if not texts:
            raise ValueError("Cannot fit tokenizer on empty text list")
            
        all_chars = set()
        for text in texts:
            if not isinstance(text, str):
                warnings.warn(f"Non-string input detected: {type(text)}. Converting to string.")
                text = str(text)
            all_chars.update(text)
        
        if not self.char2idx:
            self.char2idx = {self.pad_token: 0, self.oov_token: 1}
            self.idx2char = {0: self.pad_token, 1: self.oov_token}

            for idx, char in enumerate(sorted(all_chars), 2):
                self.char2idx[char] = idx
                self.idx2char[idx] = char

    def vocab_size(self):
        return len(self.char2idx)

class WordCharDataset(Dataset):
    """PyTorch dataset for word-level inputs with character-level targets."""
    def __init__(self, word_sequences, target_chars):
        if len(word_sequences) != len(target_chars):
            raise ValueError(f"Mismatched lengths: {len(word_sequences)} inputs vs {len(target_chars)} targets")
        if len(word_sequences) == 0:
            raise ValueError("Cannot create dataset with empty sequences")
            
        self.X = torch.tensor(word_sequences, dtype=torch.long)
        self.y = torch.tensor(target_chars, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.X):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.X)}")
        return self.X[idx], self.y[idx]
      
def segment_text(text):
    """Segment text based on language characteristics"""
    if not isinstance(text, str):
        warnings.warn(f"Non-string input detected: {type(text)}. Converting to string.")
        text = str(text)
        
    # Handle empty text
    if not text:
        return []
        
    # Check if text uses spaces
    if ' ' in text and len(text.split()) > 1:
        return text.split()
    
    # Detect script
    try:
        if re.search(r'[\u4e00-\u9fff]', text):  # Chinese characters
            # For Chinese, each character is typically treated as a token
            return list(text)
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):  # Japanese characters
            # Similar approach for Japanese
            return list(text)
        elif re.search(r'[\u0e00-\u0e7f]', text):  # Thai characters
            # Similar approach for Thai
            return list(text)
        else:
            # Default to character-level for unknown scripts without spaces
            return list(text)
    except Exception as e:
        print(f"Error in script detection: {e}. Defaulting to character-level segmentation.")
        return list(text)

class MyModel:
    """
    Word-level model that predicts the next character.
    """
    model = None
    models = dict()
    word_tokenizer = None
    word_tokenizers = dict()
    char_tokenizer = None
    char_tokenizers = dict()
    sequence_length = 15
    
    # Reduced model parameters for smaller size
    embedding_dim = 64  # Reduced from typical 200-300
    hidden_dim = 128    # Reduced from typical 256-512
    max_vocab_size = 25000  # Limit vocabulary size
    use_fp16 = True     # Use half precision (float16)

    @classmethod
    def load_training_data(cls):
        lang_to_sentences = defaultdict(list)
        
        # Define file paths with error handling
        files_to_process = [
            ('data/multi_short_100k.txt', True),  # (path, is_required)
            ('data/sentences_long_all.txt', False)
        ]
        
        pattern = r"^\d+\t([a-z]{3})\t(.+)"
        total_lines = 0
        
        for file_path, is_required in files_to_process:
            try:
                if not os.path.exists(file_path):
                    if is_required:
                        raise FileNotFoundError(f"Required file {file_path} not found")
                    else:
                        print(f"Warning: Optional file {file_path} not found, skipping")
                        continue
                
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    lines = 0
                    for line in file:
                        try:
                            match = re.match(pattern, line)
                            if match:
                                language_code = match.group(1)
                                sentence = match.group(2)
                                lowercase_sentence = sentence.lower()
                                if lowercase_sentence:  # Skip empty sentences
                                    lang_to_sentences[language_code].append(lowercase_sentence)
                                lines += 1
                        except Exception as e:
                            print(f"Error processing line in {file_path}: {e}")
                            continue
                    
                    total_lines += lines
            except Exception as e:
                if is_required:
                    raise Exception(f"Error processing required file {file_path}: {e}")
                else:
                    print(f"Warning: Error processing optional file {file_path}: {e}")
        
        if not lang_to_sentences:
            raise ValueError("No valid training data found")
            
        print(f"Loaded {total_lines} sentences for {len(lang_to_sentences)} languages.")
        return lang_to_sentences

    @classmethod
    def load_test_data(cls, fname):
        data = []
        
        # Check if file exists
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Test data file {fname} not found")
            
        try:
            with open(fname, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    # Remove newline and handle other potential issues
                    inp = line.rstrip('\n')
                    if inp:  # Skip empty lines
                        data.append(inp)
        except Exception as e:
            raise Exception(f"Error loading test data from {fname}: {e}")
            
        if not data:
            print(f"Warning: No data loaded from {fname}")
            
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        if not preds:
            raise ValueError("Cannot write empty predictions")
            
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(fname)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(fname, 'wt', encoding='utf-8') as f:
                for p in preds:
                    f.write('{}\n'.format(p))
                    
            print(f"Successfully wrote {len(preds)} predictions to {fname}")
        except Exception as e:
            raise Exception(f"Error writing predictions to {fname}: {e}")

    def estimate_model_size(self, language):
        """Estimate the size of a model in MB before saving to disk"""
        if language not in self.models:
            return 0
            
        model = self.models[language]
        word_tokenizer = self.word_tokenizers[language]
        char_tokenizer = self.char_tokenizers[language]
        
        # Calculate model parameters size
        model_params_bytes = sum(p.nelement() * (2 if self.use_fp16 else 4) for p in model.parameters())
        
        # Estimate tokenizer dictionaries size (rough approximation)
        word_tokenizer_bytes = len(word_tokenizer.word2idx) * 20  # avg 20 bytes per word entry
        char_tokenizer_bytes = len(char_tokenizer.char2idx) * 4    # avg 4 bytes per char entry
        
        # Convert to MB
        total_bytes = model_params_bytes + word_tokenizer_bytes + char_tokenizer_bytes
        total_mb = total_bytes / (1024 * 1024)
        
        return total_mb
    
    def run_train(self, data, work_dir):
        if not data:
            raise ValueError("No training data provided")
            
        total_languages = len(data)
        trained = 0
        print(f"Training models for {total_languages} languages...")
        
        # Create work directory if it doesn't exist
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            
        for language, sentences in data.items():
            if not sentences:
                print(f"Warning: No sentences for language {language}, skipping")
                continue
                
            try:
                print(f"Training model for {language} with {len(sentences)} sentences...")
                print(f"Training {trained}/{total_languages}...")
                # Limit to 50,000 sentences per language for faster training
                max_sentences = min(50000, len(sentences))
                self.train_lang_specific_model(sentences[:max_sentences], language, work_dir)
                
                # Check model size before saving
                model_size_mb = self.estimate_model_size(language)
                print(f"Estimated model size for {language}: {model_size_mb:.2f} MB")
                
                # If model is too large, reduce dimensions further and retrain
                if model_size_mb > 5.0:
                    print(f"Model for {language} exceeds 5MB limit. Further reducing dimensions...")
                    # Reduce embedding dimension more aggressively
                    old_embedding_dim = self.embedding_dim
                    old_hidden_dim = self.hidden_dim
                    old_max_vocab = self.max_vocab_size
                    
                    # Calculate scaling factor to reduce model size below 5MB
                    scaling_factor = 5.0 / model_size_mb
                    
                    # Apply scaling to parameters
                    self.embedding_dim = max(32, int(self.embedding_dim * scaling_factor))
                    self.hidden_dim = max(64, int(self.hidden_dim * scaling_factor))
                    self.max_vocab_size = max(2000, int(self.max_vocab_size * scaling_factor))
                    
                    print(f"Reducing embedding_dim: {old_embedding_dim} -> {self.embedding_dim}")
                    print(f"Reducing hidden_dim: {old_hidden_dim} -> {self.hidden_dim}")
                    print(f"Reducing vocab size: {old_max_vocab} -> {self.max_vocab_size}")
                    
                    # Retrain with reduced dimensions
                    self.train_lang_specific_model(sentences[:max_sentences], language, work_dir)
                    
                    # Verify new size
                    new_size_mb = self.estimate_model_size(language)
                    print(f"New model size for {language}: {new_size_mb:.2f} MB")
                
                self.save_specific_model(work_dir, language)
                trained += 1
                
                # Restore original dimensions for next language
                self.embedding_dim = 64
                self.hidden_dim = 128
                self.max_vocab_size = 25000
                
            except Exception as e:
                print(f"Error training model for language {language}: {e}")
                print(traceback.format_exc())
                continue
                
        if trained == 0:
            raise ValueError("No models were successfully trained")
            
        print(f"Training complete. Successfully trained {trained}/{total_languages} language models.")
    
    def train_lang_specific_model(self, data, language, work_dir, sequence_length=15):
      """
      Train a language-specific model with padding for shorter sequences.
      
      Args:
          data (list): List of sentences to train on
          language (str): Language code
          work_dir (str): Directory to save model files
          sequence_length (int): Length of word sequence for input (default: 15)
      """
      if not data:
          raise ValueError(f"No training data provided for language {language}")
      if sequence_length <= 0:
          raise ValueError(f"Sequence length must be positive, got {sequence_length}")
          
      try:
          # Create and fit word tokenizer with vocabulary size limit
          word_tokenizer = WordTokenizer(max_vocab_size=self.max_vocab_size)
          word_tokenizer.fit_on_texts(data)
          
          # Create and fit character tokenizer
          char_tokenizer = CharTokenizer()
          char_tokenizer.fit_on_texts(data)
          
          word_vocab_size = word_tokenizer.vocab_size()
          char_vocab_size = char_tokenizer.vocab_size()
          print(f"Word vocabulary size: {word_vocab_size}")
          print(f"Character vocabulary size: {char_vocab_size}")
          
          # Validation checks
          if word_vocab_size <= 2:  # Only special tokens
              raise ValueError(f"Word vocabulary too small for language {language}")
          if char_vocab_size <= 2:  # Only special tokens
              raise ValueError(f"Character vocabulary too small for language {language}")
          
          # Ensure the pad token is in the word tokenizer
          pad_token = word_tokenizer.pad_token
          if pad_token not in word_tokenizer.word2idx:
              word_tokenizer.word2idx[pad_token] = len(word_tokenizer.word2idx)
              word_tokenizer.idx2word[word_tokenizer.word2idx[pad_token]] = pad_token
              word_vocab_size += 1
          
          # Model parameters - use class variables for smaller size
          embedding_dim = self.embedding_dim
          hidden_dim = self.hidden_dim
          epochs = 5
          
          # Prepare training data
          word_sequences = []
          target_chars = []
          
          for sentence in data:
              try:
                  # Get words
                  words = segment_text(sentence)
                  
                  # Skip empty sentences
                  if not words:
                      continue
                  
                  # Pad the sequence if there aren't enough words
                  if len(words) < sequence_length + 1:
                      # Add padding to reach required length
                      padding_needed = sequence_length + 1 - len(words)
                      padded_words = [pad_token] * padding_needed + words
                      words = padded_words
                  
                  # Create sliding windows of words as input
                  for i in range(len(words) - sequence_length):
                      input_words = words[i:i + sequence_length]
                      # Target is the first character of the next word
                      next_word = words[i + sequence_length]
                      
                      if len(next_word) > 0:
                          target_char = next_word[0]
                          
                          # Convert input words to indices
                          input_indices = [word_tokenizer.word2idx.get(w, word_tokenizer.word2idx[word_tokenizer.oov_token])
                                        for w in input_words]
                          
                          # Convert target character to index
                          target_idx = char_tokenizer.char2idx.get(target_char, char_tokenizer.char2idx[char_tokenizer.oov_token])
                          
                          word_sequences.append(input_indices)
                          target_chars.append(target_idx)
              except Exception as e:
                  print(f"Error processing sentence during training: {e}")
                  continue
          
          # Check if we have training data
          if not word_sequences:
              raise ValueError(f"No valid training sequences generated for language {language}")
              
          # Create dataset and dataloader
          try:
              dataset = WordCharDataset(word_sequences, target_chars)
              dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
          except Exception as e:
              raise Exception(f"Error creating dataset: {e}")
          
          # Initialize model and move to GPU if available
          try:
              model = WordLevelCharPredictor(word_vocab_size, char_vocab_size, embedding_dim, hidden_dim, sequence_length)
              model = model.to(device)  # Move model to GPU if available
              
              # Convert model to half precision if enabled
              if self.use_fp16 and torch.cuda.is_available():
                  model = model.half()
                  print(f"Using half precision (float16) for model")
          except Exception as e:
              raise Exception(f"Error initializing model: {e}")
          
          criterion = nn.CrossEntropyLoss()
          optimizer = optim.Adam(model.parameters())
          
          # Setup gradient scaler for mixed precision training
          scaler = GradScaler() if (use_amp and torch.cuda.is_available()) else None
          
          # Training loop
          model.train()
          for epoch in range(epochs):
              epoch_loss = 0
              progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
              
              for X_batch, y_batch in progress_bar:
                  try:
                      # Move batch to device
                      X_batch = X_batch.to(device)
                      y_batch = y_batch.to(device)
                      
                      optimizer.zero_grad()
                      
                      # Use mixed precision for forward pass if enabled
                      if use_amp and torch.cuda.is_available():
                          with autocast():
                              output = model(X_batch)
                              loss = criterion(output, y_batch)
                          
                          # Use scaler for backward pass
                          scaler.scale(loss).backward()
                          scaler.step(optimizer)
                          scaler.update()
                      else:
                          # Regular training path
                          output = model(X_batch)
                          loss = criterion(output, y_batch)
                          loss.backward()
                          optimizer.step()
                      
                      epoch_loss += loss.item()
                      progress_bar.set_postfix(loss=loss.item())
                  except Exception as e:
                      print(f"Error in training batch: {e}")
                      continue
              
              print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(dataloader):.4f}")
          
          # Store model and tokenizers
          self.models[language] = model
          self.word_tokenizers[language] = word_tokenizer
          self.char_tokenizers[language] = char_tokenizer
          
      except Exception as e:
          print(f"Error training model for language {language}: {e}")
          print(traceback.format_exc())
          raise
        
    def load_models(self, work_dir):
        """
        Load all language models from a directory.
        
        Args:
            work_dir (str): Directory where models are stored
        """  
    def evaluateIfSentenceInVocab(self, sentence, tokenizer):
        """Calculate the percentage of words in the sentence that are in the vocabulary."""
        if not isinstance(sentence, str):
            warnings.warn(f"Non-string input detected: {type(sentence)}. Converting to string.")
            sentence = str(sentence)
            
        # Handle empty sentences
        if not sentence:
            return 0
            
        words = sentence.split()
        if not words:
            return 0
            
        inside_vocab = 0
        total = len(words)
        
        for word in words:
            if tokenizer.word_in_vocab(word):
                inside_vocab += 1
        
        # Return the percentage of recognized words
        return inside_vocab / total if total > 0 else 0

    def sentence_languages(self, sentence):
        """Identify the most likely language for a sentence based on vocabulary matching."""
        if not self.word_tokenizers:
            raise ValueError("No language models available. Models must be trained or loaded first.")
            
        if not isinstance(sentence, str):
            warnings.warn(f"Non-string input detected: {type(sentence)}. Converting to string.")
            sentence = str(sentence)
            
        # Handle empty sentence
        if not sentence:
            print("Warning: Empty sentence provided")
            return []
            
        language_scores = {}
        
        for language, tokenizer in self.word_tokenizers.items():
            try:
                match_score = self.evaluateIfSentenceInVocab(sentence, tokenizer)
                if match_score >= 0:  # Still maintain minimum threshold
                    language_scores[language] = match_score
            except Exception as e:
                print(f"Error evaluating sentence for language {language}: {e}")
                continue
        
        # If no languages meet the threshold, return empty list
        if not language_scores:
            print(f"No matching language found for: {sentence}")
            return []
        
        # Find the language with the highest score
        try:
            best_language = max(language_scores.items(), key=lambda x: x[1])[0]
            return [best_language]
        except Exception as e:
            print(f"Error finding best language match: {e}")
            return []

    def merge_dicts(self, dicts):
        if not dicts:
            return {}
            
        merged_dict = {}
        for d in dicts:
            if not isinstance(d, dict):
                warnings.warn(f"Expected dict but got {type(d)}. Skipping.")
                continue
                
            for key, value in d.items():
                if key in merged_dict:
                    merged_dict[key] = max(value, merged_dict[key])
                else:
                    merged_dict[key] = value
        return merged_dict
    
    def getTopChars(self, dictCharToProb):
        if not dictCharToProb:
            return "eao"  # Default fallback
            
        try:
            sorted_dict = sorted(dictCharToProb.items(), key=lambda x: x[1], reverse=True)
            top_chars = []
            for i in range(min(3, len(sorted_dict))):
                top_chars.append(sorted_dict[i][0])
            return ''.join(top_chars)
        except Exception as e:
            print(f"Error getting top characters: {e}")
            return "eao"  # Default fallback

    def run_pred(self, datain):
        if not self.models or not self.word_tokenizers or not self.char_tokenizers:
            raise ValueError("No models loaded. Train or load models before prediction.")
            
        if not datain:
            raise ValueError("No input data provided for prediction")
            
        data = []
        for line in datain:
            if not isinstance(line, str):
                warnings.warn(f"Non-string input detected: {type(line)}. Converting to string.")
                line = str(line)
            data.append(line.lower())
        
        predictions = []
        sentence_to_models = {}
        models_to_sentence = {}
        
        # Identify language for each sentence
        for sentence in data:
            try:
                models_for_sentence = self.sentence_languages(sentence)
                print(f"Sentence: {sentence}")
                print(f"Models: {models_for_sentence}") 
                
                sentence_to_models[sentence] = models_for_sentence
                for model in models_for_sentence:
                    if model in models_to_sentence:
                        models_to_sentence[model].append(sentence)
                    else:
                        models_to_sentence[model] = [sentence]
            except Exception as e:
                print(f"Error identifying language for sentence: {e}")
                # Skip problematic sentences
                sentence_to_models[sentence] = []
        
        # Make predictions for each language model
        sentence_preds = defaultdict(list)
        for model, sentences in models_to_sentence.items():
            if not sentences:
                continue
                
            try:
                if model not in self.models:
                    print(f"Warning: Model for language {model} not found")
                    continue
                    
                print(f"Model: {model}")
                print(f"Sentences: {sentences}")
                
                model_preds = self.pred_for_lang_model(
                    sentences, 
                    self.models[model], 
                    self.word_tokenizers[model], 
                    self.char_tokenizers[model]
                )
                
                for sentence, pred in model_preds.items():
                    sentence_preds[sentence].append(pred)
            except Exception as e:
                print(f"Error making predictions with model {model}: {e}")
                print(traceback.format_exc())
                continue
        
        # Merge predictions and get top characters
        top_preds_dict_version = {}
        for sentence, preds in sentence_preds.items():
            try:
                merged_preds = self.merge_dicts(preds)
                top_preds_dict_version[sentence] = self.getTopChars(merged_preds)
            except Exception as e:
                print(f"Error merging predictions for sentence: {e}")
                top_preds_dict_version[sentence] = "eao"  # Default
        
        # Create final predictions list
        for sentence in data:
            if sentence in top_preds_dict_version:
                predictions.append(top_preds_dict_version[sentence])
            else:
                # Default prediction for unknown sentences - common characters
                print(f"No prediction available for: {sentence}")
                predictions.append("eao")
        
        return predictions
    
    def pred_for_lang_model(self, data, model, word_tokenizer, char_tokenizer):
        if not model or not word_tokenizer or not char_tokenizer:
            raise ValueError("Model or tokenizers not provided")
            
        if not data:
            return {}
            
        preds = {}  # Dict to store top 3 predicted characters per sentence

        # Make sure model is in evaluation mode
        model.eval()

        for sentence in data:
            try:
                print(f"Predicting for sentence: {sentence}")
                
                if not isinstance(sentence, str):
                    warnings.warn(f"Non-string input detected: {type(sentence)}. Converting to string.")
                    sentence = str(sentence)
                    
                # Handle empty sentences
                if not sentence:
                    print(f"Empty sentence provided, using default prediction")
                    preds[sentence] = {"e": 0.5, "a": 0.3, "o": 0.2}
                    continue
                
                # Determine if language uses spaces
                has_spaces = ' ' in sentence and len(sentence.split()) > 1
                
                # Use language-appropriate tokenization
                if has_spaces:
                    words = sentence.split()
                    is_spaced_language = True
                else:
                    # For non-spaced languages, segment character by character
                    words = list(sentence)
                    is_spaced_language = False
                
                # Skip if no words
                if len(words) == 0:
                    print(f"Skipping sentence with no words: {sentence}")
                    preds[sentence] = {"e": 0.5, "a": 0.3, "o": 0.2}
                    continue
                
                # Check if the last word is potentially incomplete
                last_word = words[-1]
                last_word_in_vocab = word_tokenizer.word_in_vocab(last_word)
                
                # Get potential complete words that this could be a substring of
                potential_complete_words = []
                if not last_word_in_vocab or len(last_word) < 6:  # Check even if in vocab as it could be a substring
                    for word in word_tokenizer.word2idx.keys():
                        if word.startswith(last_word) and word != last_word:
                            potential_complete_words.append(word)
                
                # If potential completions found or word not in vocab, treat as incomplete
                is_incomplete = len(potential_complete_words) > 0 or not last_word_in_vocab
                
                # Determine context words (excluding the potentially incomplete word)
                if is_incomplete and len(words) > 1:
                    context_words = words[:-1]  # Exclude potentially incomplete word
                else:
                    context_words = words  # Use all words
                
                # Use the last few words as context
                context_words = context_words[-self.sequence_length:] if len(context_words) >= self.sequence_length else context_words
                
                # Pad if needed
                if len(context_words) < self.sequence_length:
                    context_words = [word_tokenizer.pad_token] * (self.sequence_length - len(context_words)) + context_words
                
                # Convert to indices with error handling
                word_indices = []
                for word in context_words:
                    try:
                        word_indices.append(word_tokenizer.word2idx.get(word, word_tokenizer.word2idx[word_tokenizer.oov_token]))
                    except Exception as e:
                        print(f"Error converting word to index: {e}")
                        word_indices.append(word_tokenizer.word2idx[word_tokenizer.oov_token])
                
                # Verify word indices
                if len(word_indices) != self.sequence_length:
                    print(f"Warning: Expected {self.sequence_length} indices but got {len(word_indices)}")
                    # Pad or truncate to correct size
                    if len(word_indices) < self.sequence_length:
                        word_indices = [word_tokenizer.word2idx[word_tokenizer.pad_token]] * (self.sequence_length - len(word_indices)) + word_indices
                    else:
                        word_indices = word_indices[-self.sequence_length:]
                
                # Convert to tensor and move to the appropriate device
                try:
                    input_tensor = torch.tensor([word_indices], dtype=torch.long).to(device)
                except Exception as e:
                    print(f"Error creating input tensor: {e}")
                    preds[sentence] = {"e": 0.5, "a": 0.3, "o": 0.2}
                    continue
                
                # Make prediction
                try:
                    with torch.no_grad():
                        predicted_logits = model(input_tensor)
                    
                    # Convert logits to probabilities
                    predicted_probs = F.softmax(predicted_logits, dim=1)
                except Exception as e:
                    print(f"Error in model prediction: {e}")
                    preds[sentence] = {"e": 0.5, "a": 0.3, "o": 0.2}
                    continue
                
                # Different handling based on whether the word is incomplete
                try:
                    if is_incomplete:
                        # If incomplete, filter predictions based on potential completions
                        next_char_candidates = {}
                        
                        # Get all possible next characters
                        all_chars_indices = torch.argsort(predicted_probs, descending=True)[0]
                        all_chars_with_probs = {}
                        
                        # Safely create character probability mapping
                        for idx in all_chars_indices:
                            idx_item = idx.item()
                            if idx_item in char_tokenizer.idx2char:
                                all_chars_with_probs[char_tokenizer.idx2char[idx_item]] = predicted_probs[0, idx].item()
                        
                        # Filter for next characters that continue potential completions
                        for completion in potential_complete_words:
                            next_char_position = len(last_word)
                            if next_char_position < len(completion):
                                next_char = completion[next_char_position]
                                if next_char in all_chars_with_probs:
                                    next_char_candidates[next_char] = max(
                                        next_char_candidates.get(next_char, 0),
                                        all_chars_with_probs[next_char]
                                    )
                        
                        # If we found matches based on potential completions, use those
                        if next_char_candidates:
                            top_3_chars_with_probs = dict(sorted(
                                next_char_candidates.items(), 
                                key=lambda x: x[1], 
                                reverse=True
                            )[:3])
                        else:
                            # Fall back to regular top 3 if no matches from completions
                            predicted_indices = torch.argsort(predicted_probs, descending=True)[0][:3]
                            top_3_chars_with_probs = {}
                            for idx in predicted_indices:
                                idx_item = idx.item()
                                if idx_item in char_tokenizer.idx2char:
                                    top_3_chars_with_probs[char_tokenizer.idx2char[idx_item]] = predicted_probs[0, idx].item()
                    else:
                        # If all words are complete, predict the first character of the next word
                        predicted_indices = torch.argsort(predicted_probs, descending=True)[0][:3]
                        
                        # For languages with spaces, add space as an option if predicting a new word
                        if is_spaced_language:
                            # Check if space is in the char tokenizer
                            space_idx = char_tokenizer.char2idx.get(' ', None)
                            if space_idx is not None:
                                # Make sure space has some probability in the predictions
                                top_3_chars_with_probs = {
                                    ' ': predicted_probs[0, space_idx].item()  # Add space as first prediction
                                }
                                # Add the top 2 other characters
                                counter = 0
                                for idx in predicted_indices:
                                    idx_item = idx.item()
                                    if idx_item in char_tokenizer.idx2char:
                                        char = char_tokenizer.idx2char[idx_item]
                                        if char != ' ':  # Skip space since we already added it
                                            top_3_chars_with_probs[char] = predicted_probs[0, idx].item()
                                            counter += 1
                                            if counter >= 2:  # Only get top 2 non-space chars
                                                break
                            else:
                                # Fall back to regular top 3 if space not in tokenizer
                                top_3_chars_with_probs = {}
                                for idx in predicted_indices:
                                    idx_item = idx.item()
                                    if idx_item in char_tokenizer.idx2char:
                                        top_3_chars_with_probs[char_tokenizer.idx2char[idx_item]] = predicted_probs[0, idx].item()
                        else:
                            # For non-spaced languages, use regular top 3
                            top_3_chars_with_probs = {}
                            for idx in predicted_indices:
                                idx_item = idx.item()
                                if idx_item in char_tokenizer.idx2char:
                                    top_3_chars_with_probs[char_tokenizer.idx2char[idx_item]] = predicted_probs[0, idx].item()
                    
                    # Ensure we have at least one prediction
                    if not top_3_chars_with_probs:
                        print(f"Warning: No valid characters predicted, using defaults")
                        top_3_chars_with_probs = {"e": 0.5, "a": 0.3, "o": 0.2}
                        
                    preds[sentence] = top_3_chars_with_probs
                except Exception as e:
                    print(f"Error generating character predictions: {e}")
                    preds[sentence] = {"e": 0.5, "a": 0.3, "o": 0.2}
            except Exception as e:
                print(f"Error processing sentence for prediction: {e}")
                print(traceback.format_exc())
                preds[sentence] = {"e": 0.5, "a": 0.3, "o": 0.2}

        return preds

    # Additional helper method to check if a word is a substring of other vocabulary words
    def is_substring_of_vocab_word(self, word, tokenizer):
        """Check if word is a substring of other vocabulary words"""
        if not word or not tokenizer:
            return False
            
        try:
            for vocab_word in tokenizer.word2idx.keys():
                if vocab_word.startswith(word) and vocab_word != word:
                    return True
            return False
        except Exception as e:
            print(f"Error checking substring: {e}")
            return False

    # Update the WordTokenizer class to include a method to find words starting with a prefix
    def words_with_prefix(self, prefix):
        """Return all vocabulary words that start with the given prefix"""
        if not prefix or not hasattr(self, 'word2idx'):
            return []
            
        try:
            return [word for word in self.word2idx.keys() if word.startswith(prefix) and word != prefix]
        except Exception as e:
            print(f"Error finding words with prefix: {e}")
            return []
        
    def save_specific_model(self, work_dir, language, custom_data=None):
        """
        Save a specific language model with optional custom data.
        Uses quantization and pruning to reduce model size.
        
        Args:
            work_dir (str): Directory to save model files
            language (str): Language code of the model to save
            custom_data (dict, optional): Any additional custom data to save with this model
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not language or language not in self.models:
            print(f"Error: Model for language '{language}' not found")
            return False
            
        try:
            # Create language directory
            lang_dir = os.path.join(work_dir, language)
            if not os.path.exists(lang_dir):
                os.makedirs(lang_dir)
            
            # Move model to CPU for serialization
            try:
                model = self.models[language].to('cpu')
            except Exception as e:
                print(f"Error moving model to CPU: {e}")
                model = self.models[language]  # Use as is if can't move
            
            # 1. Save model configuration with custom data
            config = {
                "word_vocab_size": self.word_tokenizers[language].vocab_size(),
                "char_vocab_size": self.char_tokenizers[language].vocab_size(),
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "sequence_length": 15,
                "use_fp16": self.use_fp16
            }
            
            # Add custom data if provided
            if custom_data is not None:
                # Ensure custom data is serializable
                try:
                    json.dumps(custom_data)
                    config["custom_data"] = custom_data
                except Exception as e:
                    print(f"Warning: Custom data not JSON serializable: {e}")
                    # Try to convert problematic values to strings
                    serializable_data = {}
                    for k, v in custom_data.items():
                        try:
                            json.dumps({k: v})
                            serializable_data[k] = v
                        except:
                            serializable_data[k] = str(v)
                    config["custom_data"] = serializable_data
                
            try:
                with open(os.path.join(lang_dir, 'config.json'), 'w') as f:
                    json.dump(config, f)
            except Exception as e:
                print(f"Error saving config: {e}")
                return False
            
            # 2. Save word tokenizer (just the mapping dictionaries)
            try:
                word_tokenizer_data = {
                    "word2idx": self.word_tokenizers[language].word2idx,
                    "idx2word": {str(k): v for k, v in self.word_tokenizers[language].idx2word.items()},
                    "oov_token": self.word_tokenizers[language].oov_token,
                    "pad_token": self.word_tokenizers[language].pad_token,
                    "max_vocab_size": self.max_vocab_size
                }
                with open(os.path.join(lang_dir, 'word_tokenizer.json'), 'w') as f:
                    json.dump(word_tokenizer_data, f)
            except Exception as e:
                print(f"Error saving word tokenizer: {e}")
                return False
            
            # 3. Save char tokenizer (just the mapping dictionaries)
            try:
                char_tokenizer_data = {
                    "char2idx": self.char_tokenizers[language].char2idx,
                    "idx2char": {str(k): v for k, v in self.char_tokenizers[language].idx2char.items()},
                    "oov_token": self.char_tokenizers[language].oov_token,
                    "pad_token": self.char_tokenizers[language].pad_token
                }
                with open(os.path.join(lang_dir, 'char_tokenizer.json'), 'w') as f:
                    json.dump(char_tokenizer_data, f)
            except Exception as e:
                print(f"Error saving char tokenizer: {e}")
                return False
            
            # 4. Save model weights - use PyTorch's built-in format with compression
            try:
                # Quantize model to int8 if not already in half precision
                if not self.use_fp16:
                    try:
                        # Convert to CPU for quantization
                        cpu_model = model.cpu()
                        # Quantize the model to 8-bit integers
                        from torch.quantization import quantize_dynamic
                        quantizable_ops = {nn.Linear, nn.LSTM}
                        quantized_model = quantize_dynamic(
                            cpu_model, qconfig_spec={op for op in quantizable_ops}, dtype=torch.qint8
                        )
                        print(f"Model quantized to 8-bit integers")
                        # Save the quantized model
                        torch.save(quantized_model.state_dict(), os.path.join(lang_dir, 'model.pt'))
                    except Exception as e:
                        print(f"Quantization failed: {e}, falling back to standard save")
                        torch.save(model.state_dict(), os.path.join(lang_dir, 'model.pt'))
                else:
                    # Model is already in fp16, just save with compression
                    torch.save(model.state_dict(), os.path.join(lang_dir, 'model.pt'))
            except Exception as e:
                print(f"Error saving model weights: {e}")
                return False
            
            # Move model back to the original device
            try:
                self.models[language] = model.to(device)
            except Exception as e:
                print(f"Error moving model back to device: {e}")
                self.models[language] = model  # Keep as CPU model if can't move
            
            # Print file information
            try:
                config_size = os.path.getsize(os.path.join(lang_dir, 'config.json')) / 1024
                word_size = os.path.getsize(os.path.join(lang_dir, 'word_tokenizer.json')) / 1024
                char_size = os.path.getsize(os.path.join(lang_dir, 'char_tokenizer.json')) / 1024
                model_size = os.path.getsize(os.path.join(lang_dir, 'model.pt')) / 1024
                total_size = config_size + word_size + char_size + model_size
                
                print(f"Model for language '{language}' saved successfully to {lang_dir}")
                print(f"File sizes (KB):")
                print(f"  config.json: {config_size:.2f}")
                print(f"  word_tokenizer.json: {word_size:.2f}")
                print(f"  char_tokenizer.json: {char_size:.2f}")
                print(f"  model.pt: {model_size:.2f}")
                print(f"  Total: {total_size:.2f}")
            except Exception as e:
                print(f"Error calculating file sizes: {e}")
            
            return True
        
        except Exception as e:
            print(f"Error saving model for language '{language}': {e}")
            print(traceback.format_exc())
            return False
      
    def load_specific_model(self, work_dir, language):
        """
        Load a specific language model.
        
        Args:
            work_dir (str): Directory where models are stored
            language (str): Language code of the model to load
        
        Returns:
            tuple: (model, word_tokenizer, char_tokenizer, custom_data) if successful,
                  None otherwise
        """
        try:
            lang_dir = os.path.join(work_dir, language)
            if not os.path.isdir(lang_dir):
                print(f"Error: Directory for language '{language}' not found")
                return None
                
            required_files = ['config.json', 'word_tokenizer.json', 'char_tokenizer.json', 'model.pt']
            if not all(os.path.exists(os.path.join(lang_dir, f)) for f in required_files):
                print(f"Error: Not all required files found for language '{language}'")
                missing = [f for f in required_files if not os.path.exists(os.path.join(lang_dir, f))]
                print(f"Missing files: {missing}")
                return None
                
            # 1. Load configuration
            try:
                with open(os.path.join(lang_dir, 'config.json'), 'r') as f:
                    config = json.load(f)
                
                # Extract custom data if it exists
                custom_data = config.get("custom_data", None)
            except Exception as e:
                print(f"Error loading config.json: {e}")
                return None
            
            # 2. Load word tokenizer
            try:
                with open(os.path.join(lang_dir, 'word_tokenizer.json'), 'r') as f:
                    word_tokenizer_data = json.load(f)
                
                word_tokenizer = WordTokenizer(
                    oov_token=word_tokenizer_data.get("oov_token", "<UNK>"),
                    pad_token=word_tokenizer_data.get("pad_token", "<PAD>")
                )
                word_tokenizer.word2idx = word_tokenizer_data.get("word2idx", {})
                word_tokenizer.idx2word = {int(k): v for k, v in word_tokenizer_data.get("idx2word", {}).items()}
                
                # Validation check
                if not word_tokenizer.word2idx:
                    raise ValueError("Word tokenizer vocabulary is empty")
            except Exception as e:
                print(f"Error loading word tokenizer: {e}")
                return None
            
            # 3. Load char tokenizer
            try:
                with open(os.path.join(lang_dir, 'char_tokenizer.json'), 'r') as f:
                    char_tokenizer_data = json.load(f)
                
                char_tokenizer = CharTokenizer(
                    oov_token=char_tokenizer_data.get("oov_token", "<UNK>"),
                    pad_token=char_tokenizer_data.get("pad_token", "<PAD>")
                )
                char_tokenizer.char2idx = char_tokenizer_data.get("char2idx", {})
                char_tokenizer.idx2char = {int(k): v for k, v in char_tokenizer_data.get("idx2char", {}).items()}
                
                # Validation check
                if not char_tokenizer.char2idx:
                    raise ValueError("Character tokenizer vocabulary is empty")
            except Exception as e:
                print(f"Error loading character tokenizer: {e}")
                return None
            
            # 4. Create and load model
            try:
                # Get required model parameters with fallbacks
                word_vocab_size = config.get("word_vocab_size", len(word_tokenizer.word2idx))
                char_vocab_size = config.get("char_vocab_size", len(char_tokenizer.char2idx))
                embedding_dim = config.get("embedding_dim", 200)
                hidden_dim = config.get("hidden_dim", 256)
                sequence_length = config.get("sequence_length", 15)
                
                # Validate parameters
                if word_vocab_size < len(word_tokenizer.word2idx):
                    print(f"Warning: Configured vocab size ({word_vocab_size}) < actual vocab size ({len(word_tokenizer.word2idx)})")
                    word_vocab_size = len(word_tokenizer.word2idx)
                    
                if char_vocab_size < len(char_tokenizer.char2idx):
                    print(f"Warning: Configured char vocab size ({char_vocab_size}) < actual char vocab size ({len(char_tokenizer.char2idx)})")
                    char_vocab_size = len(char_tokenizer.char2idx)
                
                model = WordLevelCharPredictor(
                    word_vocab_size=word_vocab_size,
                    char_vocab_size=char_vocab_size,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    sequence_length=sequence_length
                )
            except Exception as e:
                print(f"Error creating model instance: {e}")
                return None
            
            # Load state dict using PyTorch's load
            try:
                model_path = os.path.join(lang_dir, 'model.pt')
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print(traceback.format_exc())
                return None
            
            # Move model to the appropriate device
            try:
                model = model.to(device)
                model.eval()
            except Exception as e:
                print(f"Warning: Could not move model to device {device}: {e}. Using CPU.")
                model = model.to('cpu')
                model.eval()
            
            # Store in the instance
            self.models[language] = model
            self.word_tokenizers[language] = word_tokenizer
            self.char_tokenizers[language] = char_tokenizer
            
            print(f"Model for language '{language}' loaded successfully")
            
            # Return the loaded components and any custom data
            return model, word_tokenizer, char_tokenizer, custom_data
            
        except Exception as e:
            print(f"Error loading model for language '{language}': {e}")
            print(traceback.format_exc())
            return None

    # Update the existing save method to use the specific model save for each language
    def save(self, work_dir):
        """Save all language models to separate directories."""
        if not self.models:
            print("Warning: No models to save")
            return
            
        try:
            if not os.path.isdir(work_dir):
                os.makedirs(work_dir)
            
            # Save each language model separately
            success_count = 0
            total_count = len(self.models)
            
            for language in self.models.keys():
                if self.save_specific_model(work_dir, language):
                    success_count += 1
                    
            print(f"Saved {success_count}/{total_count} language models to {work_dir}")
        except Exception as e:
            print(f"Error in save operation: {e}")
            print(traceback.format_exc())

    # Update the existing load method to use the directories approach
    @classmethod
    def load(cls, work_dir):
        """Load all language models from their directories."""
        if not os.path.exists(work_dir):
            raise FileNotFoundError(f"Work directory not found: {work_dir}")
            
        instance = cls()
        
        # Check if directory is empty
        if not os.listdir(work_dir):
            print(f"Warning: Directory {work_dir} is empty")
            return instance
        
        try:
            # Iterate through directories in work_dir to find language directories
            loaded_count = 0
            for item in os.listdir(work_dir):
                lang_dir = os.path.join(work_dir, item)
                if os.path.isdir(lang_dir):
                    # Check if this is a language model directory (has all required files)
                    required_files = ['config.json', 'word_tokenizer.json', 'char_tokenizer.json', 'model.pt']
                    if all(os.path.exists(os.path.join(lang_dir, f)) for f in required_files):
                        try:
                            if instance.load_specific_model(work_dir, item) is not None:
                                loaded_count += 1
                        except Exception as e:
                            print(f"Error loading model from {lang_dir}: {e}")
                            continue
            
            if loaded_count == 0:
                print(f"Warning: No valid models found in {work_dir}")
            else:
                print(f"Successfully loaded {loaded_count} language models from {work_dir}")
                
            return instance
        except Exception as e:
            print(f"Error loading models from {work_dir}: {e}")
            print(traceback.format_exc())
            return instance

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print(test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
