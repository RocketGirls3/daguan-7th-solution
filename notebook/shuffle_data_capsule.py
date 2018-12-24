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
config = argparse.Namespace()
#########################################parameter##################
train_dir = '../input/new_data/train_set.csv'
test_dir = '../input/new_data/test_set.csv'
word2vec_dir = '../feature/word2vec_file/avito600d.w2v'
output_pro_dir = '../pro/prob_rnn_baseline4.csv'
output_dir = '../output/sub_rnn_baseline4.csv'
output_pkl_dir = '../feature/rnn.pkl'
shuffle_data_dir = '../input/new_data/shuffle_train.csv'


max_features = 800000
config.len_desc = 800000
vec_len = 600
config.maxlen = 1000
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
def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
#######################################start##########################
##############################get shuffle data#######################

if os.path.exists(shuffle_data_dir):
    train = pd.read_csv(shuffle_data_dir)
else:
    shuffle_1_data = []
    shuffle_19_data = []
    original_train = pd.read_csv(train_dir)

    label_1_text = original_train.loc[original_train['class']==1]['word_seg'].values
    label_19_text = original_train.loc[original_train['class']==19]['word_seg'].values
    for i in label_1_text:
        words = i.split(' ')
        rs=random.sample(words,len(words))
        rs = ' '.join(rs)
        shuffle_1_data.append(rs)

    for i in label_19_text:
        words = i.split(' ')
        rs=random.sample(words,len(words))
        rs = ' '.join(rs)
        shuffle_19_data.append(rs)

    df_label19 = pd.DataFrame(shuffle_19_data,columns=['word_seg'])
    df_label19['class'] = 19
    df_label19['article']='0 0 0 0'

    df_label1 = pd.DataFrame(shuffle_1_data,columns=['word_seg'])
    df_label1['class'] = 1
    df_label1['article']='0 0 0 0'

    train = pd.concat([original_train, df_label19, df_label1])
    train = train.reset_index(drop=True)

    train.to_csv(shuffle_data_dir,index=False)



###############################
print('start read train test files')
original_train = pd.read_csv(train_dir)
shuffle_data = train[original_train.shape[0]:]
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


def get_Capsule_model():
    # features_input = Input(shape=(features.shape[1],))

    inpword = Input(shape=(config.maxlen,))
    emb_word = Embedding(embedding_matrix_desc.shape[0], 600, weights=[embedding_matrix_desc], trainable=False)(inpword)
    embed_layer = SpatialDropout1D(0.4)(emb_word)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(embed_layer)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    capsule = Capsule(num_capsule=10, dim_capsule=32, routings=5, share_weights=True)(x)
    capsule = Flatten()(capsule)
    x = Dropout(0.2)(capsule)
    x = PReLU()(Dense(128)(x))

    x = Dense(19, activation="softmax")(x)
    model = Model(inputs=[inpword], outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[f1])

    return model

one_y_train=np_utils.to_categorical(Y,num_classes=19)
# one_y_val=np_utils.to_categorical(Y,num_classes=19)

from sklearn.model_selection import KFold

decay_dic = {0: 0.001, 1: 0.001, 2: 0.0009, 3: 0.0008, 4: 0.0007, 5: 0.0006, 6: 0.0005, 7: 0.0004, 8: 0.0003, 9: 0.0002,
             10: 0.0001, 11: 0.00009, 12: 0.00008, 13: 0.00007, 14: 0.00006, 15: 0.00005}

def lr_decay(epoch):
    return decay_dic[epoch]


from sklearn.model_selection import KFold



def kf_train(fold_cnt=3, rnd=1):
    now_nfold = 0
    kf = KFold(n_splits=fold_cnt, shuffle=False, random_state=233 * rnd)
    train_pred, test_pred = np.zeros((102277, 19)), np.zeros((102277, 19))
    LRDecay = LearningRateScheduler(lr_decay)
    for train_index, test_index in kf.split(original_train):
        # x,y

        now_nfold += 1
        print("now is {} fold".format(now_nfold))
        curr_x1 = tr_word_pad[train_index]
        hold_out_x1 = tr_word_pad[test_index]
        shuffle_number = 1
        shuffle_word_pad = tr_word_pad[original_train.shape[0]:]
        shuffle_word_pad_y = one_y_train[original_train.shape[0]:]
        for shuffle_train_index, shuffle_test_index in kf.split(shuffle_data):
            shuffle_train = shuffle_word_pad[shuffle_train_index]
            shuffle_train_y = shuffle_word_pad_y[shuffle_train_index]
            if shuffle_number == now_nfold:
                break
            shuffle_number += 1
        curr_x1 = np.vstack((curr_x1, shuffle_train))
        curr_y, hold_out_y = one_y_train[train_index], one_y_train[test_index]
        curr_y = np.vstack((curr_y, shuffle_train_y))

        config.batch_size = 64
        epochs = 15

        model = get_Capsule_model()

        file_path = "weights_base_cap.best.h5"
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


