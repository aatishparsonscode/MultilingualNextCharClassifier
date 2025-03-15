# MultilingualNextCharClassifier

This file contains code on how I built a Multilingual next character predictor. 
The result was 55% accuracy on 16K sentences, placing top 5/40.

# Training and Prediction Flows

## Training Flow

The training process follows these key steps:

1. **Data Loading**:
   - Loads multilingual sentences from source files
   - Organizes data by language code
   - Performs basic data cleaning and validation

2. **Model Initialization Per Language**:
   - For each language, initializes separate tokenizers and models
   - Creates a `WordTokenizer` with vocabulary size limits (25,000 words by default)
   - Creates a `CharTokenizer` to map character-level outputs

3. **Data Preparation**:
   - Segments text appropriately based on whether the language uses spaces
   - Creates sliding windows of words as input sequences
   - Each input is a sequence of words, and the target is the first character of the next word
   - Handles padding for sequences shorter than the required length

4. **Training Loop**:
   - Trains a `WordLevelCharPredictor` LSTM model
   - Uses CrossEntropyLoss and Adam optimizer
   - Implements mixed precision training when available
   - Uses a batch-based approach with progress tracking

5. **Model Optimization**:
   - Monitors model size and automatically reduces dimensions if too large
   - Applies quantization techniques to compress models under 5MB per language
   - Saves models in separate directories with configuration metadata

## Prediction Flow

The prediction process has these main components:

1. **Language Identification**:
   - For each input sentence, identifies the most likely language
   - Calculates vocabulary match scores across all trained languages
   - Maps sentences to the appropriate language model

2. **Context Processing**:
   - Analyzes whether the last word is potentially incomplete
   - Extracts context words from the input sentence
   - Creates properly padded word sequences for the model

3. **Next Character Prediction**:
   - Passes the word context through the appropriate language model
   - Generates probabilities for each possible next character
   - Implements special handling for incomplete words:
     - If the last word is incomplete, checks for possible completions
     - Filters character predictions based on potential word completions

4. **Prediction Refinement**:
   - For space-separated languages, includes spaces as a prediction option
   - Handles non-spaced languages differently (like Chinese or Japanese)
   - Merges predictions when multiple models contribute
   - Returns the top 3 most likely next characters

5. **Output Generation**:
   - Consolidates predictions for all input sentences
   - Falls back to common characters ("eao") when predictions fail
   - Formats and returns the final prediction list

This architecture allows the system to efficiently predict the next character across multiple languages while handling different language characteristics and partial words appropriately.

