# btw im working on google colab
# 1 installation of required libraries
#!pip install pandas scikit-learn mlflow joblib nltk stop-words


# 2 import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import mlflow.sklearn
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from stop_words import get_stop_words
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
np.random.seed(42)






#3 read data
df = pd.read_csv('/content/all_tickets_processed_improved_v3.csv')
print(df.head(5))  
print(df['Topic_group'].value_counts())  
print(df.shape)  
print(df.isnull().sum())  




#4 clean the data
stop_en = set(stopwords.words('english'))
stop_fr = set(get_stop_words('french'))
stop_ar = set(get_stop_words('arabic'))
all_stops = stop_en.union(stop_fr, stop_ar)

# Added more key terms for better feature retention
key_terms = {
    'facturation', 'billing', 'invoice', 'hardware', 'computer', 'laptop', 
    'password', 'login', 'access', 'storage', 'space', 'disk', 'hr', 'human', 
    'resources', 'project', 'purchase', 'buy', 'software', 'administrative',
    'rights', 'permission', 'miscellaneous', 'other', 'network', 'email'
}

def enhanced_clean_text(text):
    
    
    
    text = str(text).lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    words = text.split()
    
    processed_words = []
    for word in words:
        if len(word) < 2:  
            continue
            
        if word in key_terms:
            processed_words.append(word)
            continue
            
        if word in all_stops:
            continue
            
        # Apply stemming based on character detection
        if any(ord(char) > 127 for char in word):  # Non-ASCII (Arabic/French)
            # For Arabic/French text, use minimal processing
            processed_words.append(word)
        else:
            # For English text, apply stemming
            stemmed_word = SnowballStemmer('english').stem(word)
            processed_words.append(stemmed_word)
    
    return ' '.join(processed_words)

print("Cleaning text data...")
df['cleaned_document'] = df['Document'].apply(enhanced_clean_text)

print("\nSample of original vs cleaned text:")
sample_df = df[['Document', 'cleaned_document']].sample(3)
for idx, row in sample_df.iterrows():
    print(f"Original: {row['Document'][:100]}...")
    print(f"Cleaned: {row['cleaned_document'][:100]}...")
    print("-" * 50)



# 5 split the data i add class weight 
X = df['cleaned_document']
y = df['Topic_group']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))
print("\nClass weights:", class_weight_dict)



# 6 build the pipeline using TF-IDF and Linear SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=30000,  # Increased features e5er hal 
        ngram_range=(1, 3),  
        min_df=2,          
        max_df=0.9,        
        sublinear_tf=True   # Use sublinear TF scaling
    )),
    ('svm', LinearSVC(
        C=1.0, 
        class_weight=class_weight_dict,
        random_state=42,
        max_iter=2000       # ensure convergence
    ))
])

# Train base model
print("Training base model...")
pipeline.fit(X_train, y_train)

# Evaluate base model
y_pred_base = pipeline.predict(X_test)
accuracy_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base, average='weighted')

print("\nBase Model Performance:")
print(f"Base Accuracy: {accuracy_base:.4f}")
print(f"Base F1 Score: {f1_base:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_base))
#best accuracy 0.851 somehow


# 7 using calibratedclassifiercv approach(text classification with calibrated probabilities te5ou a9al wa9et que lo5ra )
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_svm = LinearSVC(
    C=1.0, 
    class_weight=class_weight_dict,
    random_state=42,
    max_iter=2000
)

calibrated_svc = CalibratedClassifierCV(
    estimator=base_svm,  
    cv=cv_strategy, 
    method='sigmoid'
)

pipeline_calib = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )),
    ('svm', calibrated_svc)
])

print("Training calibrated model...")
pipeline_calib.fit(X_train, y_train)

# Evaluate calibrated model
y_pred_calib = pipeline_calib.predict(X_test)
accuracy_calib = accuracy_score(y_test, y_pred_calib)
f1_calib = f1_score(y_test, y_pred_calib, average='weighted')

print("\n Calibrated Model Performance:")
print(f"Calibrated Accuracy: {accuracy_calib:.4f}")
print(f"Calibrated F1 Score: {f1_calib:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_calib))
#0.853 accuracy 5ir mel lo5ra bchwayyyaaa



# best selected (5tar 2 approach)
def evaluate_model_cv(pipeline, X, y, cv=3):
    """Evaluate model using cross-validation"""
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')
    return scores.mean(), scores.std()

# Evaluate both models with CV
print("Performing cross-validation...")
base_cv_score, base_cv_std = evaluate_model_cv(pipeline, X_train, y_train)
calib_cv_score, calib_cv_std = evaluate_model_cv(pipeline_calib, X_train, y_train)

print(f"Base Model CV F1: {base_cv_score:.4f} (±{base_cv_std:.4f})")
print(f"Calibrated Model CV F1: {calib_cv_score:.4f} (±{calib_cv_std:.4f})")

# Select best model based on CV and test performance
if calib_cv_score > base_cv_score and f1_calib > f1_base:
    best_model = pipeline_calib
    best_accuracy = accuracy_calib
    best_f1 = f1_calib
    print(" Selected: Calibrated Model")
else:
    best_model = pipeline
    best_accuracy = accuracy_base
    best_f1 = f1_base
    print(" Selected: Base Model")


# Testing for the examples
test_examples = [
    ("My computer won't start and shows blue screen", "Hardware"),
    ("I forgot my password and cannot login to system", "Access"),
    ("Problème de facturation ce mois-ci", "Administrative rights"),
    ("Need more storage space for project files", "Storage"),
    ("HR support for maternity leave application", "HR Support"),
    ("Purchase request for Adobe Creative Cloud license", "Purchase"),
    ("Internal project timeline delay reporting", "Internal Project"),
    ("مشكلة في الاتصال بالشبكة والإنترنت", "Hardware"),  # Network connection issue
    ("Request administrative rights for software installation", "Administrative rights"),
    ("General inquiry about office facilities and parking", "Miscellaneous")
]

print("Testing with 10 Example Tickets:")
print("=" * 80)

cleaned_examples = [enhanced_clean_text(example[0]) for example in test_examples]
predictions = best_model.predict(cleaned_examples)

if hasattr(best_model.named_steps['svm'], 'predict_proba'):
    probabilities = best_model.predict_proba(cleaned_examples)
else:
    decision_scores = best_model.decision_function(cleaned_examples)
    from scipy.special import softmax
    probabilities = softmax(decision_scores, axis=1)

for i, ((example, expected), pred, proba) in enumerate(zip(test_examples, predictions, probabilities)):
    confidence = max(proba) * 100
    pred_class = best_model.classes_[np.argmax(proba)]
    
    status = "CORRECT" if pred_class == expected else "WRONG"
    
    print(f"Example {i+1}: {status}")
    print(f"   Input: '{example}'")
    print(f"   Expected: {expected}")
    print(f"   Predicted: {pred_class}")
    print(f"   Confidence: {confidence:.1f}%")
    
    # Show top 3 predictions if confidence is low
    if confidence < 70:
        top_3_indices = np.argsort(proba)[-3:][::-1]
        print("   Top 3 predictions:")
        for j, idx in enumerate(top_3_indices):
            print(f"     {j+1}. {best_model.classes_[idx]}: {proba[idx]*100:.1f}%")
    
    print("-" * 60)


#8 upload model 
model_filename = 'best_topic_classification_model.pkl'
joblib.dump(best_model, model_filename)
print(f" Model saved as: {model_filename}")


# 9 dyel mlflow logging


mlflow.set_experiment("CallCenterAI_TFIDF_SVM")
with mlflow.start_run(run_name="TFIDF_SVM_Final"):
    mlflow.log_param("max_features", 30000)
    mlflow.log_param("ngram_range", (1,3))
    mlflow.log_param("C", 1.0)
    mlflow.log_param("calibration_method", "sigmoid")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metric("accuracy", best_accuracy)
    mlflow.log_metric("f1_score", best_f1)
    mlflow.sklearn.log_model(best_model, "tfidf_svm_model", registered_model_name="CallCenterAI_TFIDF_SVM") 
    print("\nMLflow Logging Complete: Model logged and registered.")





#other approach 
#  evaluate the best model
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












# aproach o5ra habit ntestiha ama te5ou wa9et 7 calibrate the model for probability estimates (more accurate probabilities)
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