(1) INSTRUCTIONS

To run load_tp2.py you need to manually download the model at:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?resourcekey=0-wjGZdNAUop6WykTtMip30g

and paste the model next to the script (same path as the script).

To run train_tp2.py you need to just execute it, if you have all the libraries installed, it will automatically do everything.
(it did for me)



(2) RUNNING TIMES

load_tp2.py:
The code took 51.23344945907593 seconds to execute.
(plot included)
(download not included)

train_tp2.py
The code took 65.12318706512451 seconds to execute.
(training and plot included)
(small download not included)



(3) PARAMETERS

For the pre-trained Word2Vec model (load_tp2.py):
- No changes were made to the default parameters of the pre-trained model.

For the trained Word2Vec model (train_tp2.py):
- vector_size: 100 (The size of the word vectors.)
- window: 5 (The maximum distance between the current and predicted word within a sentence.)
- min_count: 5 (Ignores all words with total frequency lower than this.)
- workers: [Number of CPU cores] (Used for parallel processing.)



(4) SENTENCE EMBEDDINGS

The sentence embeddings were obtained using the Word2Vec model provided by the Gensim library. 
Each sentence was first preprocessed to convert to lowercase and remove punctuation. 
The preprocessed sentence was then split into words, and for each word, its corresponding vector representation was retrieved from the Word2Vec model.

If a word was not found in the Word2Vec model's vocabulary, it was not takken into account.
The sentence embedding was obtained by averaging the vectors of all words in the sentence. 
This method ensures that each sentence is represented as a fixed-size vector.



(5) Training Corpus

For training the Word2Vec model, I choose the 'text8' dataset. 
This dataset comprises a 100MB sample of cleaned and preprocessed English Wikipedia text. 

The 'text8' dataset was easy enough to setup and train, and if required, it would have been easy to try with the larger version 'text9' dataset.