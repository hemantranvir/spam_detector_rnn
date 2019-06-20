from keras.layers import SimpleRNN, LSTM, GRU, Embedding, Dense, Flatten
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import plot_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def main(rnn_model):
    def message_to_array(msg):
        msg = msg.lower().split(' ')
        test_seq = np.array([word_index[word] for word in msg])

        test_seq = np.pad(test_seq, (500-len(test_seq), 0), 'constant', constant_values=(0))
        test_seq = test_seq.reshape(1, 500)
        return test_seq

    data = pd.read_csv("./spam_text_message_data.csv")
    print(data.head())
    print(data.tail())

    messages = []
    labels = []
    for index, row in data.iterrows():
        messages.append(row['Message'])
        if row['Category'] == 'ham':
            labels.append(0)
        else:
            labels.append(1)

    messages = np.asarray(messages)
    labels = np.asarray(labels)

    print("Number of messages: ", len(messages))
    print("Number of labels: ", len(labels))

    max_vocab = 10000
    max_len = 500

    # Ignore all words except the 10000 most common words
    tokenizer = Tokenizer(num_words=max_vocab)
    # Calculate the frequency of words
    tokenizer.fit_on_texts(messages)
    # Convert array of messages to list of sequences of integers
    sequences = tokenizer.texts_to_sequences(messages)

    # Dict keeping track of words to integer index
    word_index = tokenizer.word_index

    # Convert the array of sequences(of integers) to 2D array with padding
    # maxlen specifies the maximum length of sequence (truncated if longer, padded if shorter)
    data = pad_sequences(sequences, maxlen=max_len)

    print("data shape: ", data.shape)

    # We will use 80% of data for training & validation(80% train, 20% validation) and 20% for testing
    train_samples = int(len(messages)*0.8)

    messages_train = data[:train_samples]
    labels_train = labels[:train_samples]

    messages_test = data[train_samples:len(messages)-2]
    labels_test = labels[train_samples:len(messages)-2]

    embedding_mat_columns=32
    # Construct the SimpleRNN model
    model = Sequential()
    ## Add embedding layer to convert integer encoding to word embeddings(the model learns the
    ## embedding matrix during training), embedding matrix has max_vocab as no. of rows and chosen
    ## no. of columns
    model.add(Embedding(input_dim=max_vocab, output_dim=embedding_mat_columns, input_length=max_len))

    if rnn_model == 'SimpleRNN':
        model.add(SimpleRNN(units=embedding_mat_columns))
    elif rnn_model == 'LSTM':
        model.add(LSTM(units=embedding_mat_columns))
    else:
        model.add(GRU(units=embedding_mat_columns))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    # Training the model
    model.fit(messages_train, labels_train, epochs=10, batch_size=60, validation_split=0.2)

    # Testing the model
    pred = model.predict_classes(messages_test)
    acc = model.evaluate(messages_test, labels_test)
    print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))

    # Constructing a custom message to check model
    custom_msg = 'Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed Free entry for movies'
    test_seq = message_to_array(custom_msg)
    pred = model.predict_classes(test_seq)
    print(pred)

if __name__ == '__main__':
    main('SimpleRNN')
    main('LSTM')
    main('GRU')
