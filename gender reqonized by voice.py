!pip install python_speech_features
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from pydub import AudioSegment 
from python_speech_features import mfcc
from time import time
df=pd.read_csv('../input/common-voice/cv-valid-train.csv')
df_male=df[df['gender']=='male']
df_female=df[df['gender']=='female']
printf(df_male.shape)
print(df_female.shape)
df_male=df_male[:300]
df_female=df_female[:300]
TRAIN_PATH='../input/common-voice/cv-valid-train/'
def convert_to_wav(df,m_f,path=TRAIN_PATH):
    srcs=[]
    for file in tqdm(df['filename']):
        sound=aduiosegment.from_mp3(path+file)
        if m_f=='male':
            sound.export('male-'+file.split('/')[-1].split('.')[0]+'.wav',format='wav') 
        elif m_f=='female':
            sound.export('female-'+file.split('/')[-1].split('.')[0]+'.wav',format='wav')
    return
convert_to_wav(df_male,m_f='male')
convert_to_wav(ddf_female,m_f='female')
def ioad_audio(audio_files):
    male_voices=[]
    female_voices=[]
    for file in tqdm(audio_files):
        if file.split('-')[0]=='male':
            male_voices.append(librosa.load(file))
        elif file.split('-')[0]=='female':
            female_voices.append(librosa.load(file))
    male_voice=np.array(male_voices)
    female_voice=np.array(female_voice)
    return male_voices,female_voices
male_voices,female_voices=load_audio(os.listdir())
def extract_features(audio_data):
    audio_waves=audio_data[:,0]
    samplerate=audio_data[:,1][1]
    features=[]
    for audio_wave in tqdm(audio_waves):
        features.append(mfcc(audio_wave,samplerate,numcep=26))
    features=np.array(features)
    return features
male_features=extract_features(male_voices)
female_features=extract_features(female_voices)
def concatenate_features(audio_features):
    concatenated=audio_features[0]
    for audio_feature in tqdm(audio_features):
        concatenated=np.vstack((concatenated,audio_features))
    return concatenated
male_concatenated=concatenate_features(male_featurres)
female_concatenated=concatenate_features(female_features)
print (male_concatenated.shape)
print(female_concatenated.shape)
X=np.vstack((male_concatenated,female_concatenated))
y=np.append([0]*len(male_concatenated),[1]*len(female_concatenated))
print(X.shape)
print(y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)
clf=SVC(kernel='rbf')
start=time()
print(X_train[:50000],y_train[:50000])
print(time()-start)
start=time()
print(clf.score(X-train[:50000],y_train[:50000]))
print(time()-start)
start=time()
print(clf.score(X_test[:10000],y_test[:10000]))
print(time()-start)
svm_predictions=clf.predict(X_test[:10000])
cm=confusion_matrsvm_ix(y_test[:10000],svm_predictions)
plt.figure(figsize=(8.8))
plt.title('confusion matrix on test data')
sns.heatmap(cm,annot=True,fmt='d',cmap=plt.cm.Blues,cbar=Falsee,annot_kws={'size':14})
plt.xlabel('predicted lable')
plt.ylabel('True Label')
plt.show()
index=['SVM-RBF','SVM_poly','SVM-sigmoid','logistic regression']
values=[184.8,137.0,283.6,0.7]
plt.figure(figsize=(1,2,3))
plt.title('training duration(lower is better)')
plt.xlabel('seconds')
plt.grid(zorder=0)
for i,value in enumerate(values):
    plt.text(value+20,i,str(value)+'secs',fontsize=12,color='black',horizontalalignment='center',verticalaligment='center')
plt.show()
barWidth=0.25
index=['SVM-RBF','SVM-poly','SVM-sigmoid','logistic regression']
train_acc=[78.2,74.8,74.8,65.8]
train_acc=[76.8,74.3,74.3,65.8]
baseline=np.arrange(len(train_acc))
r1=[x+0.125 for x in baseline]
r2=[x+0.25 for x in r1]
plt.figure(figsize=(16,9))
plt.title('model performance(higher is better)')
plt.bar(r1,train_acc,width=barwidth,label='train',zorder=2)
plt.bar(r2,test_acc,width=barwidth,label='test',zorder=2)
plt.grid(zorder=0)
plt.xlabel('model')
plt.ylabel('accuracy')
plt.xticks([r+barwidth for r in range(len(train_acc))],index)
for i,value in enumerate(train_acc):
    plt.text(i+0.125,value-5,str(value),fontsize=12,color='white',horizontalalignment='center',verticalaligment='center')
for i,value in enumerate(test_acc):
    plt.text(i+0.375,value-5,str(value),fontsize=12,color='white',horizontalalignment='center',verticalaligment='center')
plt.legend()
pit.show()