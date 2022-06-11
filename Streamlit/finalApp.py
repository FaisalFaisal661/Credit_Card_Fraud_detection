import itertools
import pickle
import profile
from pyparsing import col
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from PIL import Image
import numpy as np
# import cv2
import pandas as pd
# from st_aggrid import AgGrid
# import plotly.express as px
import io
import seaborn as sns
import matplotlib.pyplot as plt



st.image('image\logo.png')
st.subheader("***Dept. Computer Science and Engineering***")
st.title(" **Title: CREDIT CARD FRAUD DETECTION** ")
st.write("""***This web Application demonstrate the detection of fraudlent credit card transation using Random Forest Algorythm***
""")



with st.sidebar:
    choose = option_menu("Menu ", ["Home", "Data description","Graphical Representation", "Run Model"],
                         icons=['house', 'kanban',
                                'bar-chart-line', 'person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

df = st.cache(pd.read_csv)('creditcard.csv')
df = df.sample(frac = 0.1, random_state = 48)
fraud = df[df.Class == 1]
valid = df[df.Class == 0]
fraud_transc_percentage = (len(fraud)/len(df))*100

if choose == 'Home':
    st.header("""**Presented by:**""")    
    st.subheader("***Faisal(1GC18CS012)***")
    st.subheader("***Fardeen Zunain(1GC18CS013)***")
    st.subheader("***Saniya Afnain(1GC18CS051)***")
    st.subheader("***Sure Naba(1GC18CS060)***")
    st.header("**Under the Guidence of:**")
    st.subheader("***Dr. Sameena Banu***")
    st.write("***Associated Professor***")

   

if choose == 'Data description':
    st.header("**CREDIT CARD FRAUD DETECTION**")
    st.write("**The Data set**", df.head())
    st.write("**Shape of DataSet**", df.shape)
    st.write("**Description of the DataSet**", df.describe())
    st.write("***Data Desription****")

if choose == 'Graphical Representation':
    #Plot 1
    vf = df["Class"].value_counts(normalize = True)    
    fig = plt.figure()
    sns.barplot(vf.index, vf*100).set_title("Percentage of Valid and Fraud Transactions")
    st.pyplot(fig)
    st.write("The number of fraud transactions are very low compared to valid transactions. We can see our dataset is highly imbalanced.")

    fig1 = plt.figure()
    sns.histplot(df["Amount"], bins=40).set_title(
        "Distribution of Monetory value feature")
    # plot 2 through graph
    st.pyplot(fig1)
    st.write("The distribution of the monetary value of all transactions is heavily right-skewed. The vast majority of transactions are relatively small and only a tiny fraction of transactions comes even close to the maximum.")
    #plot 3
    fig2 = plt.figure()
    sns.histplot(valid["Time"], bins=40).set_title(
        "Distribution of Valid transactions over Time")
    st.pyplot(fig2)
    #plot 4
    fig3 = plt.figure()
    sns.histplot(fraud["Time"], bins = 40).set_title("Distribution of Fraud transactions over Time")
    st.pyplot(fig3)
    st.write("There seems to be a decrease in number of transactions around 100000 Time mark. This might be during night. This could be the time that favours the fraudsters when valid transactions are very low.")
    #plot 5
    fig4 = plt.figure()
    sns.heatmap(df.corr(), cmap = sns.color_palette("coolwarm", as_cmap = True)).set_title("Correlation Heatmap");
    st.pyplot(fig4)
    st.write("There is not much correlation between any of the attributes")

X = df.drop(["Class"], axis = 1)
Y = df.Class

# splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

if choose == 'Run Model':
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
if choose == 'Graphical Representation':

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
   
  
   
   accuracy = model.score(X_test, Y_pred)
   'Accuracy: ', accuracy
   
   "Confusion Matrix: "
   cm = confusion_matrix(Y_test, pred)
   
   group_names = ['True Neg','True Pos','False Neg','False Pos']
   group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
   group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
   labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
   labels = np.asarray(labels).reshape(2,2)
   
   fig = plt.figure(figsize=(12,9))
   sns.heatmap(cm, annot=labels, fmt="", cmap='Reds')
   st.pyplot(fig)

if choose == 'Run Model':
    st.title("The Confusion Matrics ")
    compute_performance(model, X_train, Y_train, X_test, Y_test)


    st.write("""
    True Positive Values Shows the completly Valid or Legit Transations
    Fasle Negetive Values Show the complete Fraud Transations
    """)





