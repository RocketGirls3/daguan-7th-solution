import re
import gc
import pickle
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import *
from keras import optimizers
from keras import backend as K
from collections import Counter
from keras.utils import np_utils
from gensim.models import word2vec
from sklearn.metrics import f1_score
from keras.regularizers import l1, l2
from keras.engine import InputSpec, Layer
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text, sequence
from keras.layers import Input, Embedding, Dense
from sklearn.preprocessing import StandardScaler
from keras import optimizers, losses, activations
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from keras.layers.noise import GaussianNoise
import random
import os
import tqdm
from gensim.models.word2vec import Word2Vec
# from models_def import Attention
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from keras.callbacks import LearningRateScheduler
config = argparse.Namespace()

train_dir = '../input/new_data/train_set.csv'
test_dir = '../input/new_data/test_set.csv'
pseudo_label_dir = 'prob_rnn_baseline4.csv'
word2vec_dir = '../feature/word2vec_file/avito600d.w2v'
output_pro_dir = '../pro/prob_rnn_baseline4.csv'
output_dir = '../output/sub_rnn_baseline4.csv'
output_pkl_dir = '../feature/pseudo_rnn.pkl'
shuffle_data_dir = '../input/new_data/shuffle_train.csv'


max_features = 800000
config.len_desc = 800000
vec_len = 600
config.maxlen = 1000
count_thres = 5
config.batch_size = 64
###########
class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
###########


train = pd.read_csv(train_dir)
test = pd.read_csv(test_dir)

df_y_train = (train["class"]-1).astype(int)
test_id = test[["id"]].copy()

pseudo_label_data = pd.read_csv(pseudo_label_dir)
name = ["class_prob_%s"%i for i in range(1,20)]
pseudo_y=np.argmax(pseudo_label_data[name].values,axis=1)

all_data = pd.concat([train, test])
all_data = all_data.reset_index(drop=True)
train_offset = train.shape[0]

column = "word_seg"
tknzr_word = Tokenizer(num_words=config.len_desc)
tknzr_word.fit_on_texts(all_data[column].values)


low_count_words = [w for w, c in tknzr_word.word_counts.items() if c < count_thres]
for w in low_count_words:
    del tknzr_word.word_index[w]
    del tknzr_word.word_docs[w]
    del tknzr_word.word_counts[w]

tr_word_seq = tknzr_word.texts_to_sequences(train[column].values)
te_word_seq = tknzr_word.texts_to_sequences(test[column].values)

tr_word_pad = pad_sequences(tr_word_seq, maxlen=config.maxlen)
te_word_pad = pad_sequences(te_word_seq, maxlen=config.maxlen)

############word2vec#############
if os.path.exists(word2vec_dir):
    pass
else:
    print('start_train_word2vec')
    model = Word2Vec(size=600, window=5,max_vocab_size=500000, sg=1)
    train2 = all_data['word_seg'].values
    train2 = [text_to_word_sequence(text) for text in tqdm(train2)]
    model.build_vocab(train2)
    model.train(train2, total_examples=model.corpus_count, epochs=3)
    model.save(word2vec_dir)
    print("word2vec train done")


EMBEDDING = word2vec_dir
model = word2vec.Word2Vec.load(EMBEDDING)
word_index = tknzr_word.word_index
nb_words_desc = min(max_features, len(word_index))
embedding_matrix_desc = np.zeros((nb_words_desc+1, vec_len))
for word, i in word_index.items():
    if i >= max_features: continue
    try:
        embedding_vector = model[word]
    except KeyError:
        embedding_vector = None
    if embedding_vector is not None: embedding_matrix_desc[i] = embedding_vector
print("word2vec read done")

Y = df_y_train
pseudo_Y = pseudo_y


def get_rnn_model():
    #     features_input = Input(shape=(features.shape[1],))

    inpword = Input(shape=(config.maxlen,))
    emb_word = Embedding(embedding_matrix_desc.shape[0], 600, weights=[embedding_matrix_desc], trainable=False)(inpword)

    lDropout_titl = SpatialDropout1D(0.5)(emb_word)
    title_layer = Bidirectional(CuDNNLSTM(256, return_sequences=True))(lDropout_titl)

    title_layer = Bidirectional(CuDNNGRU(256, return_sequences=True))(title_layer)

    max_pool_til = GlobalMaxPooling1D()(title_layer)
    att = AttentionWeightedAverage()(title_layer)
    # AttentionWeightedAverage()

    all_views = concatenate([max_pool_til, att], axis=1)
    x = Dropout(0.5)(all_views)

    x = PReLU()(Dense(128)(x))

    x = Dense(19, activation="softmax")(x)
    model = Model(inputs=[inpword], outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[f1])

    return model

one_y_train=np_utils.to_categorical(df_y_train,num_classes=19)
pseudo_y_train=np_utils.to_categorical(pseudo_Y,num_classes=19)

decay_dic = {0: 0.001, 1: 0.001, 2: 0.0009, 3: 0.0008, 4: 0.0007, 5: 0.0006, 6: 0.0005, 7: 0.0004, 8: 0.0003, 9: 0.0002,
             10: 0.0001,
             11: 0.00009, 12: 0.00008, 13: 0.00007, 14: 0.00006, 15: 0.00005}


def lr_decay(epoch):
    return decay_dic[epoch]


from sklearn.model_selection import KFold

def kf_train(fold_cnt=3, rnd=1):
    now_nfold = 0
    kf = KFold(n_splits=fold_cnt, shuffle=False, random_state=233 * rnd)
    train_pred, test_pred = np.zeros((102277, 19)), np.zeros((102277, 19))
    LRDecay = LearningRateScheduler(lr_decay)
    for train_index, test_index in kf.split(train):
        now_nfold += 1
        print("now is {} fold".format(now_nfold))
        curr_x1 = tr_word_pad[train_index]
        hold_out_x1 = tr_word_pad[test_index]
        pseudo_number = 1
        #         pseudo_word_pad = tr_word_pad[original_train.shape[0]:]
        #         pseudo_word_pad_y = one_y_train[original_train.shape[0]:]
        for pseudo_train_index, pseudo_test_index in kf.split(te_word_pad):
            pseudo_train = te_word_pad[pseudo_train_index]
            pseudo_train_y = pseudo_y_train[pseudo_train_index]
            if pseudo_number == now_nfold:
                break
            pseudo_number += 1
        curr_x1 = np.vstack((curr_x1, pseudo_train))
        curr_y, hold_out_y = one_y_train[train_index], one_y_train[test_index]
        curr_y = np.vstack((curr_y, pseudo_train_y))

        config.batch_size = 64
        epochs = 15

        model = get_rnn_model()

        file_path = "weights_base.best.h5"
        checkpoint = ModelCheckpoint(file_path, save_best_only=True, verbose=1, monitor='val_f1', mode='max')
        early = EarlyStopping(monitor='val_f1', mode='max', patience=2, )
        callbacks_list = [checkpoint, early, LRDecay]

        model.fit(curr_x1, curr_y,
                  batch_size=config.batch_size, epochs=epochs,
                  validation_data=(hold_out_x1, hold_out_y),
                  callbacks=callbacks_list)

        model.load_weights(file_path)

        y_test = model.predict(te_word_pad)
        test_pred += y_test
        hold_out_pred = model.predict(hold_out_x1)
        train_pred[test_index] = hold_out_pred

        # clear
        del model
        gc.collect()
        K.clear_session()
    test_pred = test_pred / fold_cnt
    print('-------------------------------')
    try:
        print('all eval', sqrt(mean_squared_error(Y, train_pred)))
    finally:
        return train_pred, test_pred


print('def done')

train_pred,test_pred = kf_train(fold_cnt=10,rnd=4)

with open(output_pkl_dir,'wb') as fout:
    pickle.dump([train_pred,test_pred],fout)