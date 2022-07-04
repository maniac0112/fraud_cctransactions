import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

cc_data= pd.read_csv("creditcard.csv") #converting the csv dataset to pandas dataframe for processing

if (cc_data["Class"].value_counts()[0]>cc_data["Class"].value_counts()[1]): #very less anomalous class dataset

    true_ta=cc_data[cc_data.Class==0]
    false_ta=cc_data[cc_data.Class==1]
    balanced_true_ta=true_ta.sample(n=cc_data["Class"].value_counts()[1]) #limited data of class "0" to not overwhelm the class "1"
    dataset=pd.concat([balanced_true_ta,false_ta], axis=0) #axis=0 means rowwise concatenation
    X=dataset.drop(columns="Class", axis=1) #axis=1 implied dropping column
    Y=dataset["Class"] 

    trainX,testX,trainY,testY=train_test_split(X,Y, test_size=0.25, stratify=Y) #splitting the dataset for training and testing
    model=LogisticRegression() #initializing our logistic regression model
    model.fit(trainX,trainY)   #training our model using training dataset
    #evaluation:
    X_train_prediction=model.predict(trainX)
    training_accuracy= accuracy_score(X_train_prediction,trainY) #training dataset accuracy

    X_test_prediction=model.predict(testX)
    test_accuracy= accuracy_score(X_test_prediction,testY) #test dataset accuracy
    testY=testY.to_numpy() #converting pd dataframe to np array
    tp=0
    fp=0
    fn=0
    for i in range(X_test_prediction.shape[0]):
        t_val= testY.item(i)
        prediction=X_test_prediction.item(i)
        if (prediction==1 and t_val==1):
            tp+=1
        if (prediction==0 and t_val==1):
            fn+=1
        if (prediction==1 and t_val==0):
            fp+=1
    
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    F1=2*precision*recall/(precision+recall)
        

    print("Training accuracy is: "+ str(training_accuracy))
    print("Test accuracy is: "+ str(test_accuracy))
    print("Precision, recall and F1 scores are: ")
    print(precision,recall,F1)
    print("respectively")