--------------------------------------
1. Instructions for Running the Script
--------------------------------------

To run the script, follow these steps:

a. Run the script using the following command:
   python finetune_tp4.py

b. if all the required libraries are available, it should download the pre-trained BERT model on its own (it did for me)

c. as it executes, it trains a model for each language and saves it in a new folder for each language

d. after each training, it tests the model as well 

e. in the second execution of the code, if the models are saved correctly, it will just test, it won´t train a new one

---------------------------------
2. List of All Pre-trained Models
---------------------------------

For our script, we utilize the following pre-trained model:

- BERT-Base Multilingual Cased (bert-base-multilingual-cased)
  - Type: Transformer-based model designed for understanding multiple languages
  - Number of Parameters: Approximately 178 million
  - Description: This version of BERT is trained on a large corpus of text in 104 languages. It's well-suited for tasks that require understanding and processing text in multiple languages. This model uses a cased version which takes letter case into account, potentially providing more precise understanding of the text.

Note: This is the primary model used across various languages for our NLP tasks. The model's multilingual capabilities enable it to understand and process text in a wide range of languages with a single pre-trained model.

