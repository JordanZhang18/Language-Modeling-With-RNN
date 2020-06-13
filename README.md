# Language Modeling With RNN
 NLP with Disaster Tweets competition on Kaggle, using two RNN models

 Training and testing data were read as dataframes. Only columns ['text', 'target'] were used as training data.
 The training data contains 7613 rows and no null values. I did some research and found some of the labels were not correct. Therefore, I relabeled some training data. label.value_counts() shows the training data contains 4380 false (0) and 3233 true instances.

 I also defined a few functions to clean the texts: removing html links, url, @, emogis, as well as some stop words such as ‘a’ or ‘the’. These elements are likely to become noises. I also used stemmer.stem function to shorten and standardize some words to the stems. For example, deeds --> deed and residents --> resid.
 Both training and testing data are encoded to tokens. Using  keras.preprocessing.text.Tokenizer and padding functions, the texts are converted to arrays of lists containing integers. The sentences were pre-padded to maximum length of 28 elements.

 Review research design and modeling methods

 The first model I built contains: one embedding layer, with the dictionary size as input and contain 100 nodes. A simple RNN layer, containing 100 nodes with return sequence = True. A flatten layer and lastly a dense layer of 1 node using sigmoid as activation, as the label is binary.

 The loss function for fitting is binary cross entropy and I used ‘adam’ as optimizer, with initial learning rate of 0.0001. Two callback functions were used: reduce LR on plateau with patience =2 and early stopping with patience =5. The metric to monitor is loss.

 The second model I built contains: one same embedding layer. A bidirectional LSTM layer with 128 nodes. And lastly a dense layer of 1 node using sigmoid as activation. The loss function and callbacks are consistent with model 1.

 Review results, evaluate models
 
 The first model was trained 50 epoches and early stopped at 49 epoch. Achieving loss=0.023 and validation loss=1.066. The second model was trained 50 epoches, achieving loss=0.0317 and validation loss=0.982.
 The encoded test data was used to make prediction and the result shows model 1 yield f1 score 0.745, and model 2 f1=0.753. The bidirectional LSTM is slightly better than the simple RNN model.
