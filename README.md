## Simulation With K-fold NLP

This project was created to test different machine learning and deep learning methods for natural language processing using different ways for attribute extractions in texts written in Portuguese language.

## List with Different ways to extract attributes:

1. LIWC2007_Portugues
2. Word2Vec Using buscape model
3. Bertimbau base and large (neuralmind/bert-large-portuguese-cased, neuralmind/bert-base-portuguese-cased)
4. Bert base and large (bert-base-uncased, bert-large-uncased)
5. Bert_pierreguillou (pierreguillou/bert-base-cased-squad-v1.1-portuguese)
6. Roberta in versions: base, large, josu version, cardiffnlp and cardiffnlp sentiment (xlm-roberta-base, xlm-roberta-large, josu/roberta-pt-br, cardiffnlp/twitter-xlm-roberta-base, cardiffnlp/twitter-xlm-roberta-base-sentiment)


## Paht config asserts
All files to use was in asserts path
1. asserts\datasets -> This path must be your dataset.
2. asserts\datasets_tratados -> This path will be your dataset after make spelling corrections, remove symbols..
3. asserts\modelos -> This path should have all the model language you use and not download with python
4. asserts\resultados -> This path will be your results alfer finished processing.
5. asserts\rv_dataset -> This path is used for app to save your dataset after apply model language  
6. asserts\rv_models -> This path is used for app to save model language generated (using only for word2vec)
