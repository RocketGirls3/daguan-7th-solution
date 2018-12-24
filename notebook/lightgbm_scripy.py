import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from scipy import sparse
from keras.utils import np_utils
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
import random

train_dir = '../input/new_data/train_set.csv'
shuffle_data_dir = '../input/new_data/shuffle_train.csv'
trn_term_doc_path = '../feature/text_tifi/trn_term_doc2.npz'
trn_term_word_path = '../feature/text_tifi/trn_term_word2.npz'
trn_term_word_three_path = '../feature/text_tifi/trn_term_word_three2.npz'
trn_term_word_four_path ='../feature/text_tifi/trn_term_word_four2.npz'
trn_char ='../feature/text_tifi/trn_char2.npz'

test_term_doc_path = '../feature/text_tifi/test_term_doc2.npz'
test_term_word_path = '../feature/text_tifi/test_term_word2.npz'
test_term_word_three_path ='../feature/text_tifi/test_term_word_three2.npz'
test_term_word_four_path = '../feature/text_tifi/test_term_word_four2.npz'
test_char ='../feature/text_tifi/test_char2.npz'

pkl_path = '../feature/stackint_pkl_file/ligbm.pkl'
out_put = '../output/lgb_baseline3.csv'
pro_output = '../pro/prob_lgb_baseline3.csv'
#############################################################
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

column = "word_seg"
# train = pd.read_csv('../input/new_data/train_set.csv')
test = pd.read_csv('../input/new_data/test_set.csv')
test_id = test["id"].copy()
y_all = (pd.Series(train['class'])-1).astype(int)

if os.path.exists(trn_term_doc_path):
    vec_doc = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=400000,analyzer='word')
    trn_term_doc = vec_doc.fit_transform(train[column])
    test_term_doc = vec_doc.transform(test[column])

    sparse.save_npz(trn_term_doc_path, trn_term_doc)
    sparse.save_npz(test_term_doc_path, test_term_doc)
else:
    trn_term_doc = sparse.load_npz(trn_term_doc_path)
    test_term_doc = sparse.load_npz(test_term_doc_path)

if os.path.exists(trn_term_word_path):
    vec_doc =  TfidfVectorizer(ngram_range=(1,1),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=400000,analyzer='word')
    trn_term_word = vec_doc.fit_transform(train[column])
    test_term_word = vec_doc.transform(test[column])

    sparse.save_npz(trn_term_word_path, trn_term_word)
    sparse.save_npz(test_term_word_path, test_term_word)
else:
    trn_term_word = sparse.load_npz(trn_term_word_path)
    test_term_word = sparse.load_npz(test_term_word_path)

if os.path.exists(trn_term_word_three_path):
    vec_doc = TfidfVectorizer(ngram_range=(3,3),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=400000,analyzer='word')
    trn_term_word_three = vec_doc.fit_transform(train[column])
    test_term_word_three = vec_doc.transform(test[column])

    sparse.save_npz(trn_term_word_three_path, trn_term_word_three)
    sparse.save_npz(test_term_word_three_path, test_term_word_three)
else:
    trn_term_word_three = sparse.load_npz(trn_term_word_three_path)
    test_term_word_three = sparse.load_npz(test_term_word_three_path)

if os.path.exists(trn_term_doc_path):
    vec_doc = TfidfVectorizer(ngram_range=(4,4),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=400000,analyzer='word')
    trn_term_word_four = vec_doc.fit_transform(train[column])
    test_term_word_four = vec_doc.transform(test[column])

    sparse.save_npz(trn_term_word_four_path, trn_term_word_four)
    sparse.save_npz(test_term_word_four_path, test_term_word_four)
else:
    trn_term_word_four = sparse.load_npz(trn_term_word_four_path)
    test_term_word_four = sparse.load_npz(test_term_word_four_path)

if os.path.exists(trn_term_doc_path):
    vec_doc =TfidfVectorizer(ngram_range=(1,2),min_df=3,use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=10000,analyzer='char')
    trn_char = vec_doc.fit_transform(train[column])
    test_char = vec_doc.transform(test[column])

    sparse.save_npz(trn_char, trn_char)
    sparse.save_npz(test_char, test_char)
else:
    trn_char = sparse.load_npz(trn_char)
    test_char = sparse.load_npz(test_char)

print("特征提取完成")
docLen  =pd.DataFrame( np.array(train[column].map(lambda x : len(x.split(" ")))/39759).reshape(-1,1))
doclen_word = pd.DataFrame( np.array(train['article'].map(lambda x : len(x.split(" ")))/train[column].map(lambda x : len(x.split(" ")))).reshape(-1,1))
docLen_test  =pd.DataFrame( np.array(test[column].map(lambda x : len(x.split(" ")))/39759).reshape(-1,1))
doclen_word_test = pd.DataFrame( np.array(test['article'].map(lambda x : len(x.split(" ")))/test[column].map(lambda x : len(x.split(" ")))).reshape(-1,1))

train_feaure =  hstack([trn_term_doc, trn_term_word, trn_term_word_three, trn_term_word_four, trn_char, doclen_word]).tocsr()
del trn_term_doc, trn_term_word, trn_term_word_three, trn_term_word_four,trn_char,doclen_word
test_festure =  hstack([test_term_doc,test_term_word,test_term_word_three,test_term_word_four,test_char,doclen_word_test]).tocsr()
del test_term_doc,test_term_word,test_term_word_three,test_term_word_four,test_char,doclen_word_test
gc.collect()

test_prob = pd.DataFrame([],columns=["class_prob_%s"%i for i in range(1,20)])
train_prob = pd.DataFrame([],columns=["class_prob_%s"%i for i in range(1,20)])
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat, average='macro'), True


one_y_train=np_utils.to_categorical(y_all,num_classes=20)

for i in range(19):
    print(i)
    train_target = one_y_train[:,i]
#     model = LogisticRegression(C=6, solver='sag')
#     sfm = SelectFromModel(model, threshold=0.2)
#     print(train_feaure.shape)
#     train_sparse_matrix = sfm.fit_transform(train_feaure, train_target)
#     print(train_sparse_matrix.shape)
    train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_feaure, train_target, test_size=0.05, random_state=144)
#     test_sparse_matrix = sfm.transform(test_festure)
    d_train = lgb.Dataset(train_sparse_matrix, label=y_train)
    d_valid = lgb.Dataset(valid_sparse_matrix, label=y_valid)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.1,
              'application': 'binary',
              'num_leaves': 100,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 2,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.6,
              'nthread': 4,}
    model = lgb.train(params,
                      feval=lgb_f1_score,
                      train_set=d_train,
                      num_boost_round=2000,
                      valid_sets=watchlist,
                      verbose_eval=10,
                      early_stopping_rounds=200)
    train_prob['class_prob_%s'%(i+1)] = model.predict(train_feaure)
    test_prob['class_prob_%s'%(i+1)] = model.predict(test_festure)

with open(pkl_path,'wb') as fout:
    pickle.dump([train_prob,test_prob],fout)

test_prob["id"]=list(test_id)
test_prob.to_csv(pro_output,index=None)

preds=np.argmax(test_prob[name].values,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv(output,index=None)
t2=time.time()
print("time use:",t2-t1)