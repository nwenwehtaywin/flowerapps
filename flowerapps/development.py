from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X,y=load_iris(return_X_y=True,as_frame=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
log=LogisticRegression()
log.fit(X_train,y_train)
test_result=log.predict(X_test)
train_result=log.predict(X_train)

print(accuracy_score(train_result,y_train)*100)
print(accuracy_score(test_result,y_test)*100)

#Save model
import pickle
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(log, f)

print("Model saving is done.")