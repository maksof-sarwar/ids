import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import time 
  
# reading features list 
f=open("./project/kddcup.names.csv", 'r')
print(f.read()) 

cols ="""duration, 
protocol_type, 
service, 
flag, 
src_bytes, 
dst_bytes, 
land, 
wrong_fragment, 
urgent, 
hot, 
num_failed_logins, 
logged_in, 
num_compromised, 
root_shell, 
su_attempted, 
num_root, 
num_file_creations, 
num_shells, 
num_access_files, 
num_outbound_cmds, 
is_host_login, 
is_guest_login, 
count, 
srv_count, 
serror_rate, 
srv_serror_rate, 
rerror_rate, 
srv_rerror_rate, 
same_srv_rate, 
diff_srv_rate, 
srv_diff_host_rate, 
dst_host_count, 
dst_host_srv_count, 
dst_host_same_srv_rate, 
dst_host_diff_srv_rate, 
dst_host_same_src_port_rate, 
dst_host_srv_diff_host_rate, 
dst_host_serror_rate, 
dst_host_srv_serror_rate, 
dst_host_rerror_rate, 
dst_host_srv_rerror_rate"""

columns =[] 
for c in cols.split(', '): 
    if(c.strip()): 
       columns.append(c.strip()) 
  
columns.append('target') 
print(len(columns))

# reading attacks types
f=open("./project/training_attack_types.csv", 'r') 
print(f.read())



attacks_types = { 
    'normal': 'normal', 
'back': 'dos', 
'buffer_overflow': 'u2r', 
'ftp_write': 'r2l', 
'guess_passwd': 'r2l', 
'imap': 'r2l', 
'ipsweep': 'probe', 
'land': 'dos', 
'loadmodule': 'u2r', 
'multihop': 'r2l', 
'neptune': 'dos', 
'nmap': 'probe', 
'perl': 'u2r', 
'phf': 'r2l', 
'pod': 'dos', 
'portsweep': 'probe', 
'rootkit': 'u2r', 
'satan': 'probe', 
'smurf': 'dos', 
'spy': 'r2l', 
'teardrop': 'dos', 
'warezclient': 'r2l', 
'warezmaster': 'r2l', 
} 



path = "./project/kddcup.data_10_percent.gz"
df = pd.read_csv(path, names = columns)
df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]]) 
df.head()
df.shape

df.isnull().sum()


# Finding categorical features 
num_cols = df._get_numeric_data().columns 
  
cate_cols = list(set(df.columns)-set(num_cols)) 
cate_cols.remove('target') 
cate_cols.remove('Attack Type') 
  
cate_cols

df = df.dropna('columns')# drop columns with NaN 
  
df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values 
  
corr = df.corr() 
  
plt.figure(figsize =(15, 12)) 
  
sns.heatmap(corr) 
  
plt.show()
#######################################################################################################################
# This variable is highly correlated with num_compromised and should be ignored for analysis. 
#(Correlation = 0.9938277978738366) 
df.drop('num_root', axis = 1, inplace = True) 
  
# This variable is highly correlated with serror_rate and should be ignored for analysis. 
#(Correlation = 0.9983615072725952) 
df.drop('srv_serror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9947309539817937) 
df.drop('srv_rerror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with srv_serror_rate and should be ignored for analysis. 
#(Correlation = 0.9993041091850098) 
df.drop('dst_host_srv_serror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9869947924956001) 
df.drop('dst_host_serror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with srv_rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9821663427308375) 
df.drop('dst_host_rerror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9851995540751249) 
df.drop('dst_host_srv_rerror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with srv_rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9865705438845669) 
df.drop('dst_host_same_srv_rate', axis = 1, inplace = True)


# protocol_type feature mapping 
pmap = {'icmp':0, 'tcp':1, 'udp':2} 
df['protocol_type'] = df['protocol_type'].map(pmap)

# flag feature mapping 
fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10} 
df['flag'] = df['flag'].map(fmap)

df.drop('service', axis = 1, inplace = True)




################################################################################################################
# step 2

##  Code: Importing libraries and splitting the dataset

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

# Splitting the dataset 
df = df.drop(['target', ], axis = 1) 
print(df.shape) 
  
# Target variable and train set 
y = df[['Attack Type']] 
X = df.drop(['Attack Type', ], axis = 1) 
  
sc = MinMaxScaler() 
X = sc.fit_transform(X) 
  
# Split test and train data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42) 
print(X_train.shape, X_test.shape) 
print(y_train.shape, y_test.shape) 



# Gaussian Naive Bayes 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 
  
clfg = GaussianNB() 
start_time = time.time() 
clfg.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time)
start_time = time.time() 
y_test_pred = clfg.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time)
print("Train score is:", clfg.score(X_train, y_train)) 
print("Test score is:", clfg.score(X_test, y_test))
# Decision Tree  
from sklearn.tree import DecisionTreeClassifier 
  
clfd = DecisionTreeClassifier(criterion ="entropy", max_depth = 4) 
start_time = time.time() 
clfd.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time)
start_time = time.time() 
y_test_pred = clfd.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time)
print("Train score is:", clfd.score(X_train, y_train)) 
print("Test score is:", clfd.score(X_test, y_test))
from sklearn.ensemble import RandomForestClassifier 
  
clfr = RandomForestClassifier(n_estimators = 30) 
start_time = time.time() 
clfr.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time)
start_time = time.time() 
y_test_pred = clfr.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time)
print("Train score is:", clfr.score(X_train, y_train)) 
print("Test score is:", clfr.score(X_test, y_test))



  
from sklearn.svm import SVC 
  
clfs = SVC(gamma = 'scale') 
start_time = time.time() 
clfs.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time)

  
start_time = time.time() 
y_test_pred = clfs.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time) 
print("Train score is:", clfs.score(X_train, y_train)) 
print("Test score is:", clfs.score(X_test, y_test))
#########################################################################################################

from sklearn.linear_model import LogisticRegression 
  
clfl = LogisticRegression(max_iter = 120) 
start_time = time.time() 
clfl.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time)
start_time = time.time() 
y_test_pred = clfl.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time)
print("Train score is:", clfl.score(X_train, y_train)) 
print("Test score is:", clfl.score(X_test, y_test))


from sklearn.ensemble import GradientBoostingClassifier 
  
clfg = GradientBoostingClassifier(random_state = 0) 
start_time = time.time() 
clfg.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time)
start_time = time.time() 
y_test_pred = clfg.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time)
print("Train score is:", clfg.score(X_train, y_train)) 
print("Test score is:", clfg.score(X_test, y_test))

names = ['NB', 'DT', 'RF', 'SVM', 'LR', 'GB'] 
values = [87.951, 99.058, 99.997, 99.875, 99.352, 99.793] 
f = plt.figure(figsize =(15, 3), num = 10) 
#plt.subplot(131) 
#plt.bar(names, values)

names = ['NB', 'DT', 'RF', 'SVM', 'LR', 'GB'] 
values = [87.903, 99.052, 99.969, 99.879, 99.352, 99.771] 
f = plt.figure(figsize =(15, 3), num = 10) 
#plt.subplot(131) 
#plt.bar(names, values)

names = ['NB', 'DT', 'RF', 'SVM', 'LR', 'GB'] 
values = [1.11452, 2.44087, 17.08491, 218.26840, 92.94222, 633.229] 
f = plt.figure(figsize =(15, 3), num = 10) 
#plt.subplot(131) 
#plt.bar(names, values)

names = ['NB', 'DT', 'RF', 'SVM', 'LR', 'GB'] 
values = [1.54329, 0.14877, 0.199471, 126.50875, 0.09605, 2.95039] 
f = plt.figure(figsize =(15, 3), num = 10) 
#plt.subplot(131) 
#plt.bar(names, values) 
     
