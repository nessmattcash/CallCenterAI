# btw im working on google colab
# 1 installation of required libraries
#!pip install pandas scikit-learn mlflow joblib nltk stop-words


# 2 import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from stop_words import get_stop_words
nltk.download('stopwords')
nltk.download('punkt')






#3 read data
df = pd.read_csv('/content/all_tickets_processed_improved_v3.csv')
print(df.head(5))  
print(df['Topic_group'].value_counts())  
print(df.shape)  
print(df.isnull().sum())  




#4 clean the data
stop_en = set(stopwords.words('english'))
stop_fr = set(get_stop_words('french'))
all_stops = stop_en.union(stop_fr)
stemmer_en = SnowballStemmer('english')
stemmer_fr = SnowballStemmer('french')

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())  
    text = re.sub(r'\d+', '', text)  
    words = text.split()
    words = [stemmer_en.stem(word) if word in stop_en else stemmer_fr.stem(word) for word in words if word not in all_stops]
    return ' '.join(words)

df['cleaned_document'] = df['Document'].apply(clean_text)
print(df[['Document', 'cleaned_document']].sample(5))  # Examples



# 5 split the data
X = df['cleaned_document']
y = df['Topic_group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, y_train.value_counts(normalize=True))  # the basic 0.8 0.2 split 


# 6 build the pipeline using TF-IDF and Linear SVM
pipeline = Pipeline([ 
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('svm', LinearSVC(C=1))
])
pipeline.fit(X_train, y_train)
y_pred_base = pipeline.predict(X_test)
print("Base Accuracy:", accuracy_score(y_test, y_pred_base))
print(classification_report(y_test, y_pred_base))
#best accuracy 0.85 somehow


# 7 calibrate the model for probability estimates (more accurate probabilities)
calibrated_svc = CalibratedClassifierCV(LinearSVC(), cv=5, method='sigmoid')  # For probs
pipeline_calib = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', calibrated_svc)
])


# 8 hyperparameter tuning with GridSearchCV
param_grid = {
    'tfidf__max_features': [5000, 10000, 15000],
    'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
    'tfidf__min_df': [1, 2, 5],
    'svm__estimator__C': [0.1, 1, 10], 
    'svm__estimator__class_weight': [None, 'balanced']  
}
cv = StratifiedKFold(n_splits=5)  
grid_search = GridSearchCV(pipeline_calib, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best Params:", grid_search.best_params_)
print("Best CV F1:", grid_search.best_score_)   # te5ou wa9et kbir


# 9 evaluate the best model
best_model = grid_search.best_estimator_ 
y_pred = best_model.predict(X_test)  
y_proba = best_model.predict_proba(X_test)  
print("Test Accuracy:", accuracy_score(y_test, y_pred))  
print("Test F1:", f1_score(y_test, y_pred, average='weighted'))  
print(classification_report(y_test, y_pred))  # enchallah 0.87+ mazelt majarabtech 

#10 fake testing 
sample = ["My bill is wrong and I can't login"]  
cleaned = [clean_text(s) for s in sample]  
pred = best_model.predict(cleaned)[0]  
proba = max(best_model.predict_proba(cleaned)[0])  
print(f"Category: {pred}, Confidence: {proba:.2f}")


# 11 dyel mlflow logging

mlflow.set_experiment("TFIDF_SVM_Perfect")
with mlflow.start_run():
    mlflow.log_params(grid_search.best_params_) 
    mlflow.log_metrics({"test_acc": accuracy_score(y_test, y_pred), "test_f1": f1_score(y_test, y_pred, average='weighted')})  # Save scores
    mlflow.sklearn.log_model(best_model, "model")  # Save the model



# 12 telechargi ken cv 
joblib.dump(best_model, 'tfidf_svm_perfect.pkl') 