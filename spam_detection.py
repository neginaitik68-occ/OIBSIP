import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv("spam.csv",encoding = 'latin-1')

#printing first five lines on data
print(data.head())

#selecting useful data
print(data.columns)

data = data[['v1','v2']]
print(data.head()) 

#renaming the labels

data.columns = ['label','message']
print(data.head())

#checking unique variables

print(data['label'].unique())

#converting labels to numbers

data['label'] = data['label'].map({
    'ham':0,
    'spam':1
})
print(data.head())

#verify

print(data['label'].unique())

# seperate features and target

x = data['message']
y = data['label']

print(x.head())
print(y.head())

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x)
print(x.shape)

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print(x_train.shape)
print(x_test.shape)
#creating model
model = MultinomialNB()
model.fit(x_train,y_train)

#make predictions
y_pred = model.predict(x_test)

#calculation of accuracy

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)
#testing new email
email = ["Congratulations! You have won a free iPhone. Click here now!"]
email_vector = vectorizer.transform(email)
prediction = model.predict(email_vector)
if prediction[0] == 1:
    print("spam email")
else:
    print("not spam")    


cm = confusion_matrix(y_test,y_pred)
print(cm)

print(classification_report(y_test,y_pred))