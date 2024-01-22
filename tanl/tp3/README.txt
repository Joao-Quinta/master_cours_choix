(1)
just run the code, for it to work you need to have all the models for spacy downloaded, and the data inputs in the correct path
data/language/

(2)
time for running spacy_tp3.py --> Running time of the script: 51.04 seconds
time for running crf_tp3.py --> Running time of the script: 22.52 seconds

(3)
In this project, I utilized specific spaCy models for each language, such as pt_core_news_md for Portuguese and en_core_web_sm for English,
to perform Named Entity Recognition. Each model was loaded using spaCy's load function.
The test data was processed through these models, and then the output was converted to the BIO format for evaluation.