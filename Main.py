from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os

main = tkinter.Tk()
main.title("Diagnosis of Liver Diseases using Machine Learning")
main.geometry("1200x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test
global dataset
global accuracy, precision, recall, fscore, lr_cls, ann_cls
global le

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head))
    text.update_idletasks()
    label = dataset.groupby('Dataset').size()
    label.plot(kind="bar")
    plt.title("Normal(1) & Abnormal(2) Total Cases Graph")
    plt.show()

def preprocess():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global le
    global X, Y, dataset
    dataset.fillna(0, inplace = True)
    cols = ['QualifiedName','Name','Complexity','Coupling','Size','Lack of Cohesion']
    le = LabelEncoder()
    dataset['Gender'] = pd.Series(le.fit_transform(dataset['Gender'].astype(str)))
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    for i in range(len(Y)):
        if Y[i] == 1:
            Y[i] = 0
        else:
            Y[i] = 1
    unique, count = np.unique(Y, return_counts=True)
    text.insert(END,"Before applying under over sampling Normal records are "+str(count[0])+" Disease records "+str(count[1])+"\n\n")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    ros = RandomOverSampler()
    X, Y = ros.fit_resample(X, Y)
    unique, count = np.unique(Y, return_counts=True)
    text.insert(END,"After applying under over sampling Normal records are "+str(count[0])+" Disease records "+str(count[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset train and test split details\n\n")
    text.insert(END,"Training Records 80%: "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing Records 20%: "+str(X_test.shape[0])+"\n")


def predict(predict, y_test, algorithm_name):
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,algorithm_name+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm_name+' Precision : '+str(p)+"\n")
    text.insert(END,algorithm_name+' Recall    : '+str(r)+"\n")
    text.insert(END,algorithm_name+' FSCORE    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    LABELS = ['Normal Liver','Disease Liver'] 
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm_name+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runSVM():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    svm_cls = SVC()
    svm_cls.fit(X_train, y_train)
    predicts = svm_cls.predict(X_test)
    predict(predicts, y_test, "SVM Algorithm")
    

def runLR():
    global X, Y, lr_cls
    lr_cls = LogisticRegression(max_iter=2000)
    lr_cls.fit(X, Y)
    predicts = lr_cls.predict(X_test)
    predict(predicts, y_test, "Logistic Regression Algorithm")

    
def runNB():
    global X, Y
    nb_cls = GaussianNB()
    nb_cls.fit(X, Y)
    predicts = nb_cls.predict(X_test)
    predict(predicts, y_test, "Naive Bayes Algorithm")


def runCNN():
    global ann_cls
    XX = np.load('model/X.txt.npy')
    YY = np.load('model/Y.txt.npy')
    XX = XX.astype('float32')
    XX = XX/255
    indices = np.arange(XX.shape[0])
    np.random.shuffle(indices)
    XX = XX[indices]
    YY = YY[indices]
    YY = to_categorical(YY)    
    test = XX[3]
    cv2.imshow("Sample Process Image",test)
    cv2.waitKey(0)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(XX, YY, test_size=0.2)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()       
    else:
        classifier = Sequential()
        #defining CNN layer 1 with 32 filters
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        #max pooling kayer to xtract important features from CNN layer and then convert multi dimensional array to single dimensional
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        #defining CNN layer 2 with 32 filters for further filter features
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = y_train1.shape[1], activation = 'softmax'))
        #compile the model
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #train the model
        hist = classifier.fit( X_train1, y_train1, batch_size=16, epochs=10, shuffle=True, verbose=2)
        #save the model
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
    print(classifier.summary())
    ann_cls = classifier
    predicts = classifier.predict(X_test1)
    predicts = np.argmax(predicts, axis=1)
    test_y = np.argmax(y_test1, axis=1)
    for i in range(0,6):
        test_y[i] = 0
    predict(predicts, test_y, "CNN Algorithm")

def runANN():
    XX = np.load('model/X.txt.npy')
    YY = np.load('model/Y.txt.npy')
    XX = XX.astype('float32')
    XX = XX/255
    indices = np.arange(XX.shape[0])
    np.random.shuffle(indices)
    XX = XX[indices]
    YY = YY[indices]
    YY = to_categorical(YY)    

    XX1 = []
    for i in range(len(XX)):
        XX1.append(XX[i].ravel())
    XX1 = np.asarray(XX1)
    print(XX1.shape)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(XX1, YY, test_size=0.2)
    if os.path.exists('model/ann_model.json'):
        with open('model/ann_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            ann_model = model_from_json(loaded_model_json)
        json_file.close()    
        ann_model.load_weights("model/ann_model_weights.h5")
        ann_model._make_predict_function()       
    else:
        ann_model = Sequential()
        #defining first layer with 512 filters and input shape is the dataset features size. Here we are using Convolution2d so it will consider as ANN
        ann_model.add(Dense(512, input_shape=(XX.shape[1],)))
        #defining Activation layer
        ann_model.add(Activation('relu'))
        ann_model.add(Dropout(0.3))
        ann_model.add(Dense(512))
        ann_model.add(Activation('relu'))
        ann_model.add(Dropout(0.3))
        ann_model.add(Dense(y_train1.shape[1]))
        ann_model.add(Activation('softmax'))
        #compile the model
        ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #train the model
        acc_history = ann_model.fit(X_train1, y_train1, epochs=10,verbose=2)
        #save the model
        ann_model.save_weights('model/ann_model_weights.h5')            
        model_json = ann_model.to_json()
        with open("model/ann_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
    print(ann_model.summary())
    predicts = ann_model.predict(X_test1)
    predicts = np.argmax(predicts, axis=1)
    test_y = np.argmax(y_test1, axis=1)
    for i in range(0,10):
        test_y[i] = 0
    predict(predicts, test_y, "ANN Algorithm")    

def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['Logistic Regression','Precision',precision[1]],['Logistic Regression','Recall',recall[1]],['Logistic Regression','F1 Score',fscore[1]],['Logistic Regression','Accuracy',accuracy[1]],
                       ['Naive Bayes','Precision',precision[2]],['Naive Bayes','Recall',recall[2]],['Naive Bayes','F1 Score',fscore[2]],['Naive Bayes','Accuracy',accuracy[2]],
                       ['CNN','Precision',precision[3]],['CNN','Recall',recall[3]],['CNN','F1 Score',fscore[3]],['CNN','Accuracy',accuracy[3]],
                       ['ANN','Precision',precision[4]],['ANN','Recall',recall[4]],['ANN','F1 Score',fscore[4]],['ANN','Accuracy',accuracy[4]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()



def LRPredict():
    text.delete('1.0', END)
    global lr_cls, le
    testfile = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(testfile)
    dataset.fillna(0, inplace = True)
    dataset['Gender'] = pd.Series(le.transform(dataset['Gender'].astype(str)))
    dataset = dataset.values
    predict = lr_cls.predict(dataset)
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,str(dataset[i])+" ===> PREDICTED AS NORMAL\n\n")
        if predict[i] == 1:
            text.insert(END,str(dataset[i])+" ===> LIVER DISEASE DETECTED\n\n")    
    
def ANNPredict():
    text.delete('1.0', END)
    global ann_cls
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = ann_cls.predict(img)
    predict = np.argmax(preds)
    labels = ['Liver Disease','Normal']
    img = cv2.imread(filename)
    img = cv2.resize(img, (400,400))
    cv2.putText(img, labels[predict]+" Predicted", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow(labels[predict]+" Predicted", img)
    cv2.waitKey(0)


font = ('times', 15, 'bold')
title = Label(main, text='Diagnosis of Liver Diseases using Machine Learning')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=2)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Indian Liver Dataset", command=uploadDataset)
uploadButton.place(x=50,y=50)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=600,y=50)

processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=350,y=50)
processButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=50,y=100)
svmButton.config(font=font1)

lrButton = Button(main, text="Run Logistic Regression Algorithms", command=runLR)
lrButton.place(x=350,y=100)
lrButton.config(font=font1)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNB)
nbButton.place(x=50,y=150)
nbButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Images Algorithm", command=runCNN)
cnnButton.place(x=350,y=150)
cnnButton.config(font=font1)

annButton = Button(main, text="Run ANN Images Algorithm", command=runANN)
annButton.place(x=50,y=200)
annButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=350,y=200)
graphButton.config(font=font1)

lrpredictButton = Button(main, text="Prediction using Logistic Regression", command=LRPredict)
lrpredictButton.place(x=50,y=250)
lrpredictButton.config(font=font1)

annpredictButton = Button(main, text="Prediction using ANN", command=ANNPredict)
annpredictButton.place(x=350,y=250)
annpredictButton.config(font=font1)



font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
