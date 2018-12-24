import re
import gc
import time
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
from keras import initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import LearningRateScheduler
import random

import os
import tqdm
from gensim.models.word2vec import Word2Vec

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

config = argparse.Namespace()
#########################################parameter##################
train_dir = '../input/new_data/train_set.csv'
test_dir = '../input/new_data/test_set.csv'
word2vec_dir = '../feature/word2vec_file/avito600d.w2v'
output_pro_dir = '../pro/prob_dpcnn_2000.csv'
output_dir = '../output/sub_dpcnn_2000.csv'
output_pkl_dir = '../feature/dpcnn2000.pkl'


max_features = 800000
config.len_desc = 800000
vec_len = 600
config.maxlen = 2000
count_thres = 5
config.batch_size = 64

#######################################function#######################
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
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
#######################################start##########################

###############################
print('start read train test files')
train = pd.read_csv(train_dir)
test = pd.read_csv(test_dir)
all_data = pd.concat([train, test])
all_data = all_data.reset_index(drop=True)
train_offset = train.shape[0]
all_data['word_len'] = all_data['word_seg'].apply(len)
all_data['word_unique'] = all_data['word_seg'].apply(lambda comment: len(set( w for w in comment.split())))
all_data['word_unique_vs_len'] = all_data['word_unique'] /  all_data['word_len']
train = all_data.iloc[:train_offset,:]
test = all_data.iloc[train_offset:,:]
from sklearn.preprocessing import StandardScaler
features = train['word_unique_vs_len'].fillna(0)
test_features = test['word_unique_vs_len'].fillna(0)
features = features.reshape(-1, 1)
test_features = test_features.reshape(-1, 1)
ss = StandardScaler()
ss.fit(np.vstack((features, test_features)))
features = ss.transform(features)
test_features = ss.transform(test_features)

df_y_train = (train["class"]-1).astype(int)
test_id = test[["id"]].copy()

print ('pre_processed done')

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

tr_word_post = pad_sequences(tr_word_seq, maxlen=config.maxlen, padding='post', truncating='post')

te_word_post = pad_sequences(te_word_seq, maxlen=config.maxlen, padding='post', truncating='post')

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


def get_model():
    filter_nr = 128
    filter_size = 3
    max_pool_size = 3
    max_pool_strides = 2
    dense_nr = 256
    spatial_dropout = 0.4
    dense_dropout = 0.4

    inpword = Input(shape=(2000,))
    inparticle = Input(shape=(2000,))

    emb_comment1 = Embedding(embedding_matrix_desc.shape[0], 600, weights=[embedding_matrix_desc], trainable=False)(
        inpword)
    #     emb_comment1 = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix],trainable=False)(inpword)
    emb_comment1 = SpatialDropout1D(spatial_dropout)(emb_comment1)

    block11 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(emb_comment1)
    block11 = BatchNormalization()(block11)
    block11 = PReLU()(block11)
    block11 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block11)
    block11 = BatchNormalization()(block11)
    block11 = PReLU()(block11)

    resize_emb1 = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(emb_comment1)
    resize_emb1 = PReLU()(resize_emb1)

    block1_output1 = add([block11, resize_emb1])
    block1_output1 = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output1)

    block21 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1_output1)
    block21 = BatchNormalization()(block21)
    block21 = PReLU()(block21)
    block21 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block21)
    block21 = BatchNormalization()(block21)
    block21 = PReLU()(block21)

    block2_output1 = add([block21, block1_output1])
    block2_output1 = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output1)

    block31 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2_output1)
    block31 = BatchNormalization()(block31)
    block31 = PReLU()(block31)
    block31 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block31)
    block31 = BatchNormalization()(block31)
    block31 = PReLU()(block31)

    block3_output1 = add([block31, block2_output1])
    block3_output1 = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output1)

    block41 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3_output1)
    block41 = BatchNormalization()(block41)
    block41 = PReLU()(block41)
    block41 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block41)
    block41 = BatchNormalization()(block41)
    block41 = PReLU()(block41)

    output1 = add([block41, block3_output1])
    output1 = GlobalMaxPooling1D()(output1)

    emb_comment2 = Embedding(embedding_matrix_desc.shape[0], 600, weights=[embedding_matrix_desc], trainable=False)(
        inparticle)
    emb_comment2 = SpatialDropout1D(spatial_dropout)(emb_comment2)

    block12 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(emb_comment2)
    block12 = BatchNormalization()(block12)
    block12 = PReLU()(block12)
    block12 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block12)
    block12 = BatchNormalization()(block12)
    block12 = PReLU()(block12)

    resize_emb2 = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(emb_comment2)
    resize_emb2 = PReLU()(resize_emb2)

    block1_output2 = add([block12, resize_emb2])
    block1_output2 = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output2)

    block22 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1_output2)
    block22 = BatchNormalization()(block22)
    block22 = PReLU()(block22)
    block22 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block22)
    block22 = BatchNormalization()(block22)
    block22 = PReLU()(block22)

    block2_output2 = add([block22, block1_output2])
    block2_output2 = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output2)

    block32 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2_output2)
    block32 = BatchNormalization()(block32)
    block32 = PReLU()(block32)
    block32 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block32)
    block32 = BatchNormalization()(block32)
    block32 = PReLU()(block32)

    block3_output2 = add([block32, block2_output2])
    block3_output2 = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output2)

    block42 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3_output2)
    block42 = BatchNormalization()(block42)
    block42 = PReLU()(block42)
    block42 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block42)
    block42 = BatchNormalization()(block42)
    block42 = PReLU()(block42)

    output2 = add([block42, block3_output2])
    output2 = GlobalMaxPooling1D()(output2)

    output = concatenate([output1, output2])
    output = Dense(dense_nr, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(dense_dropout)(output)
    x = Dense(19, activation="softmax")(output)
    model = Model(inputs=[inpword, inparticle], outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[f1])

    return model

one_y_train=np_utils.to_categorical(Y,num_classes=19)
one_y_val=np_utils.to_categorical(Y,num_classes=19)

from sklearn.model_selection import KFold

decay_dic = {0:0.001, 1:0.001, 2:0.0009, 3:0.0008, 4:0.0007, 5:0.0006, 6:0.0005, 7:0.0004, 8:0.0003, 9:0.0002, 10:0.0001,
            11:0.00009, 12:0.00008, 13:0.00007, 14:0.00006, 15:0.00005}
def lr_decay(epoch):
    return decay_dic[epoch]
from sklearn.model_selection import KFold
def kf_train(fold_cnt=3, rnd=1):
    now_nfold = 0
    kf = KFold(n_splits=fold_cnt, shuffle=False, random_state=233 * rnd)
    train_pred, test_pred = np.zeros((102277, 19)), np.zeros((102277, 19))
    LRDecay = LearningRateScheduler(lr_decay)
    for train_index, test_index in kf.split(train):
        # x,y
        now_nfold += 1
        print("now is {} fold".format(now_nfold))

        curr_x1, curr_x2 = tr_word_pad[train_index], tr_word_post[train_index]
        hold_out_x1, hold_out_x2 = tr_word_pad[test_index], tr_word_post[test_index]
#         curr_x1 = tr_word_pad[train_index]
#         hold_out_x1 = tr_word_pad[test_index]
        curr_y, hold_out_y = one_y_train[train_index], one_y_train[test_index]

#         kfold_X_features = features[train_index]
#         kfold_X_valid_features = features[test_index]

        config.batch_size=64
        epochs = 15

        model = get_model()

        file_path = "weights_base_dpcnn_2000.best.h5"
        checkpoint = ModelCheckpoint(file_path, save_best_only=True, verbose=1, monitor='val_f1', mode='max')
        early = EarlyStopping(monitor='val_f1', mode='max', patience=2, )
        callbacks_list = [checkpoint, early, LRDecay]

        model.fit([curr_x1, curr_x2], curr_y,
                  batch_size=config.batch_size, epochs=epochs,
                  validation_data=([hold_out_x1,hold_out_x2], hold_out_y),
                  callbacks=callbacks_list)


        model.load_weights(file_path)

        y_test = model.predict([te_word_pad, te_word_post])
        test_pred += y_test
        hold_out_pred = model.predict([hold_out_x1, hold_out_x2])
        train_pred[test_index] = hold_out_pred

        del model
        gc.collect()
        K.clear_session()
    test_pred = test_pred / fold_cnt
    print('-------------------------------')
    try:
        print('all eval', sqrt(mean_squared_error(Y, train_pred)))
    finally:
        return train_pred, test_pred

model_time = time.time()
print("start")
train_pred,test_pred = kf_train(fold_cnt=10,rnd=4)
print (train_pred.shape,test_pred.shape)
print ("[{}] finished nn model".format((time.time()-model_time)/3600))

with open(output_pkl_dir,'wb') as fout:
    pickle.dump([train_pred,test_pred],fout)

val_result = np.argmax(train_pred,axis=1)
print ('score is {}'.format(f1_score(Y, val_result, average='macro')))


test_prob=pd.DataFrame(test_pred)
test_prob.columns=["class_prob_%s"%i for i in range(1,test_pred.shape[1]+1)]
test_prob["id"]=list(test_id["id"])
test_prob.to_csv(output_pro_dir ,index=None)


preds=np.argmax(test_pred,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv(output_dir,index=None)



