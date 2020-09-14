
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import ExtraTreesClassifier as ETC
import xgboost as xgb
import re


# In[2]:


def give_first(s):
    return s.split(",")[0]


# In[3]:


def pred_img(df,Model_rf,Scl):
    if ((len(df['spoofed'])>0) | (len(df['document_sentiment'])>0) | (len(df['sentence_sentiment'])>0)|(len(df['logos'])>0)):
        df['spoofed'] = np.random.choice(['UNKNOWN'], df.shape[0])
        df['document_sentiment'] = np.random.choice([0], df.shape[0])
        df['sentence_sentiment'] = np.random.choice([0], df.shape[0])
        df['logos']=np.random.choice([0], df.shape[0])
    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0']
    if 'index' in df.columns:
        del df['index']
    X= preprocess(df)[0]
    lab = ['document_sentiment', 'LIKELY', 'POSSIBLE', 'UNKNOWN', 'UNLIKELY','VERY_LIKELY', 'VERY_UNLIKELY', 'Min_score', 'Diff_score','Label_vect_mean', 'Label_vect_sum', 'top_websites_vect_mean','top_websites_vect_sum']
    for col in lab:
        if(col in X.columns):
            continue
        else:
            X[col] = 0
    X = Scl.transform(X)
    print(df)
    return(Model_rf.predict(X))


# In[4]:


def fill_na_doc(df):
    df_temp=df['document_sentiment'].isna()
    df['document_sentiment'][((df_temp==True) & (df.propoganda==1))]=-0.1
    df['document_sentiment'][((df_temp==True) & (df.propoganda==0))]= 0.1
    return(df)


# In[5]:


def fill_na_sent(df):
    df_temp=df['sentence_sentiment'].isna()
    df['sentence_sentiment'][((df_temp==True) & (df.propoganda==1))]=-0.1
    df['sentence_sentiment'][((df_temp==True) & (df.propoganda==0))]= 0.1
    return(df)


# In[6]:


def logo_trim(df):
    df['logos']=df['logos'].astype(str)
    df['logos']=df['logos'].astype(str).str.strip("[")
    df['logos']=df['logos'].astype(str).str.strip("]")
    df['logos']=df['logos'].apply(lambda x: give_first(x))
    df['logos']=df['logos'].astype(str).str.strip("'")

    df['logos']=df['logos'].astype(str).str.replace('\[|\]|\'', '')
    df['logos']=df['logos'].replace('', 'No_logo')

    df['logos']=df['logos'].replace('None', 'No_logo')
    df['logos']=df['logos'].replace('NATO Umbrella', 'NATO')
    df['logos']=df['logos'].replace('North Atlantic Treaty Organization (NATO)', 'NATO')

    df['logos']=df['logos'].replace('Russia Today', 'Russia')
    df['logos']=df['logos'].replace('Russian Towers', 'Russia')
    return(df)


# In[7]:


def Min_score_and_Diff_score(df):
    buf2=0
    buf1=0
    buf1_list=[]
    buf2_list=[]
    for k in list(df['sentence_sentiment']):
        if k:
            buf1=np.array(k).max()-np.array(k).min()
            buf2=np.array(k).min()
            buf1_list.append(buf1)
            buf2_list.append(buf2)
        else:
            buf1_list.append(0)
            buf2_list.append(0)
            continue
    df['Min_score']=buf2_list
    df['Diff_score']=buf1_list
    return(df)


# In[8]:


def labels_check(df):
    main_list=[]
    df.best_search_label=df.best_search_label.str.replace('[^a-zA-Z]', ' ')
    for line in df.best_search_label:
        line=re.sub("\s+", ",", line.strip())
        main_list.append(line)
    df.best_search_label=main_list
    return(df)


# In[9]:


def preprocess(df):
    df=df.iloc[1:,:]
#     df.set_index('img_id',inplace=True)
    del df['img_id']
    if 'index' in df.columns:
        df.drop(['index'],inplace=True,axis=1)

    if 'logos' in df.columns:
        df=logo_trim(df) 
    
    df.dropna(inplace=True)
    
    df['spoofed']=df['spoofed'].astype(str).str.replace('\[|\]|\'', '')
    one_hot_encoded= pd.get_dummies(df['spoofed'])
    df.drop(['spoofed'],axis=1,inplace=True)
    df=pd.concat([df,one_hot_encoded],sort=False,axis=1)
    
    
    df['top_websites']=df['top_websites'].astype(str).str.replace('\[|\]|\'', '')
    df['best_search_label']=df['best_search_label'].astype(str).str.replace('\[|\]|\'', '')

    
    df=Min_score_and_Diff_score(df)
    df.drop(['sentence_sentiment'],axis=1,inplace=True)
    df=labels_check(df)
    labels=np.array(df['best_search_label'].astype(str))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(labels)
    X.toarray().mean(axis=1)
    df['Label_vect_mean']=X.toarray().mean(axis=1)
    df['Label_vect_sum']=X.toarray().sum(axis=1)

    
    ls1=[]
    for k in list(df['top_websites']):
        if k!="":
            ls1.append([i.split(".")[-1] for i in k.split(",")])
        else:
            ls1.append('')
    df['top_websites']=ls1
    URL=np.array(df['top_websites'].astype(str))
    X = vectorizer.fit_transform(URL)
    X.toarray().mean(axis=1)
    df['top_websites_vect_mean']=X.toarray().mean(axis=1)
    df['top_websites_vect_sum']=X.toarray().sum(axis=1)
    
    
    df.drop(['best_search_label'],axis=1, inplace=True)
    df.drop(['top_websites'],axis=1,inplace=True)
    df.drop(['logos'],axis=1,inplace=True)
    
    if 'propoganda' in df.columns:
        y=df['propoganda']
        X=df.drop(['propoganda'],axis=1)
    else:
        X=df
        y=0
    return(X,y)


# In[10]:


def Model_rec(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    scaler = MinMaxScaler()
    column_names=X_train.columns.values
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_train=pd.DataFrame(data=X_train,columns=column_names)
    X_test=pd.DataFrame(data=X_test,columns=column_names)
    forest = ETC(n_estimators=250,max_depth=10 , random_state=np.random)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)
    return(forest,scaler,X_test,y_test)




