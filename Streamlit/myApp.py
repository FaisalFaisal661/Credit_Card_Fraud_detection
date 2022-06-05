from pyexpat import model
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
import timeit
import pickle

from traitlets import default

# Title
st.title("""Credit Card Fraud Detection""")


# reading CSV file from source
df =st.cache(pd.read_csv)("creditcard.csv")
df = df.sample(frac = 0.1, random_state = 48)


#exploring data
if st.sidebar.checkbox("Show information of  DataSet"):
    st.write("**The Data set**",df.head())
    st.write("**Shape of DataSet**",df.shape)
    st.write("**Description of the DataSet**",df.describe())

#Exploring if missing Values    
if st.sidebar.checkbox("Show missing values"):
    st.write(df.isna().sum()) 

#seperating fraud and Legit Transation in the current data setdd
fraud = df[df.Class == 1]
valid = df[df.Class == 0]
fraud_transc_percentage = (len(fraud)/len(df))*100


#Printing Fraud vs Legit Transation
if st.sidebar.checkbox("Statistics of Fraud and Legit Transations"):
    st.write("** Total Transations **")
    st.write("Total Transactions: ", len(df))
    st.write("Valid Transactions: ", len(valid))
    st.write("Fraud Transactions: ", len(fraud))
    st.write("Percentage of Fraud transcations: %.3f%%" %fraud_transc_percentage)


if st.sidebar.checkbox("Show plot 1"):
    vf = df["Class"].value_counts(normalize = True)    
    fig = plt.figure()
    sns.barplot(vf.index, vf*100).set_title("Percentage of Valid and Fraud Transactions")
    st.pyplot(fig)
    st.write("The number of fraud transactions are very low compared to valid transactions. We can see our dataset is highly imbalanced.")

if st.sidebar.checkbox("Show Plots 2"):
    fig1 = plt.figure()
    sns.histplot(df["Amount"], bins = 40).set_title("Distribution of Monetory value feature")
    st.pyplot(fig1)
    st.write("The distribution of the monetary value of all transactions is heavily right-skewed. The vast majority of transactions are relatively small and only a tiny fraction of transactions comes even close to the maximum.")

if st.sidebar.checkbox("Show Plots 3"):
    fig2 = plt.figure()
    sns.histplot(valid["Time"], bins = 40).set_title("Distribution of Valid transactions over Time")
    st.pyplot(fig2)
    
if st.sidebar.checkbox("Show Plots 4"):
    fig3 = plt.figure()
    sns.histplot(fraud["Time"], bins = 40).set_title("Distribution of Fraud transactions over Time")
    st.pyplot(fig3)
    st.write("There seems to be a decrease in number of transactions around 100000 Time mark. This might be during night. This could be the time that favours the fraudsters when valid transactions are very low.")
    
if st.sidebar.checkbox("Show Plots 5"):
    fig4 = plt.figure()
    sns.heatmap(df.corr(), cmap = sns.color_palette("coolwarm", as_cmap = True)).set_title("Correlation Heatmap");
    st.pyplot(fig4)
    st.write("There is not much correlation between any of the attributes")

# obtaining X and Y
X = df.drop(["Class"], axis = 1)
Y = df.Class

# splitting data into training and testing sets
from sklearn.model_selection import train_test_split, cross_val_score
size = st.sidebar.slider("Test set size", min_value = 0.2, max_value = 0.4,value=0.3)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = size, random_state = 0)

# shape of train and test data
if st.sidebar.checkbox("Show shape of train and test data"):
    st.write("Shape of X_train: ", X_train.shape)
    st.write("Shape of Y_train: ", Y_train.shape)
    st.write("Shape of X_test: ", X_test.shape)
    st.write("Shape of Y_test: ", Y_test.shape)

#importimg our model from pickle
model = pickle.load(open('random_forest.plk', 'rb'))
pred = model.predict(X_test)

def feature_sort(model, X_train, Y_train):
    # feature selection
    mod = model
    # fit model
    mod.fit(X_train, Y_train)
    imp = mod.feature_importances_
    return imp
importance = feature_sort(model, X_train, Y_train)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

if st.sidebar.checkbox("Show plot of feature importance"):
    fig = plt.figure()
    sns.barplot([x for x in range(len(importance))], importance).set_title("Feature Importance")
    plt.xlabel("Variable Number")
    plt.ylabel("Importance")
    st.pyplot(fig)

#from Jupyter notebook

def plot_cm(cm, classes, normalize = False, title = 'Confusion matrix',cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized Confusion matrix")
    else:
        print("Confusion matrix without normalization")
    print(cm)
    
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    threah = cm.max()/2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), 
                horizontalalignment='center',
                 color = 'white' if cm[i, j] > threah else "black" )
    
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()



def compute_performance(model, X_train, Y_train, X_test, Y_test):
   model.fit(X_train, Y_train)
   Y_pred = model.predict(X_test)
   
  
   
   accuracy = accuracy_score(Y_test, Y_pred)
   'Accuracy: ', accuracy
   
   "Confusion Matrix: "
   cm = confusion_matrix(Y_test, Y_pred)
   
   group_names = ['False Neg','False Pos','True Neg','True Pos']
   group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
   group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
   labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
   labels = np.asarray(labels).reshape(2,2)
   
   fig = plt.figure()
   sns.heatmap(cm, annot=labels, fmt="", cmap='Reds')
   st.pyplot(fig)


if st.sidebar.checkbox("Run a Random Forest classifier model"):
    st.title("The Conussion Matrics ")
    classifier =model
    imbalance_rect = 'no Retifier'
    compute_performance(model, X_train, Y_train, X_test, Y_test)
    cnf_matrix = confusion_matrix(Y_test, pred)
    plot_cm(cnf_matrix , classes=[0,1])

    st.write("""
    False negetive Values Shows the completly Valid or Legit Transations
    True Positive Values Show the complete Fraud Transations
    """)