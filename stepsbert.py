#1 installation 
#!pip install torch transformers datasets accelerate evaluate huggingface-hub mlflow joblib pandas scikit-learn nltk stop-words sentencepiece

#2 imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
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
import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm  
import gc  


nltk.download('stopwords')
nltk.download('punkt')
np.random.seed(42)
torch.manual_seed(42)

torch.cuda.empty_cache()
gc.collect()


#3 visualization 
# Load data
df = pd.read_csv('/content/all_tickets_processed_improved_v3.csv')
print("First 5 rows:")
print(df.head(5))
print("\nCategory distribution:")
print(df['Topic_group'].value_counts())
print("\nDataset shape:", df.shape)
print("\nMissing values:", df.isnull().sum())

# data analysis for me to understand 
print("\n" + "="*60)
print("PERFECTED DATA ANALYSIS")
print("="*60)
print(f"Total samples: {len(df):,}")
print(f"Number of classes: {df['Topic_group'].nunique()}")
print(f"Average text length: {df['Document'].str.len().mean():.0f} characters")
print(f"Text length std: {df['Document'].str.len().std():.0f} characters")

class_dist = df['Topic_group'].value_counts()
print(f"\nClass imbalance ratio: {class_dist.max()/class_dist.min():.1f}x")

# Plot 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
class_dist.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Class Distribution')
ax1.set_xticklabels(class_dist.index, rotation=45, ha='right')

# Pie chart
colors = plt.cm.Set3(np.linspace(0, 1, len(class_dist)))
ax2.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax2.set_title('Class Proportions')

plt.tight_layout()
plt.show()

df['text_length'] = df['Document'].str.len()
print(f"\nText Length Statistics:")
print(f"Min: {df['text_length'].min()} chars")
print(f"Max: {df['text_length'].max()} chars")
print(f"95th percentile: {df['text_length'].quantile(0.95):.0f} chars")
#graphs y3awnouk bech tchouf distribution


#4 cleaning function
stop_en = set(stopwords.words('english'))
stop_fr = set(get_stop_words('french'))
stop_ar = set(get_stop_words('arabic'))
all_stops = stop_en.union(stop_fr, stop_ar)

# patterns
domain_patterns = {
    'hardware': [
        r'\b(laptop|computer|desktop|printer|monitor|keyboard|mouse|device|equipment|machine|workstation|screen|display|cable|port|technical|repair|replace|broken|damaged|not working|issue|hardware)\b',
        r'\b(pc|mac|imprimante|ordinateur|écran|clavier|souris|dispositif)\b',
        r'\b(كمبيوتر|لاب توب|طابعة|شاشة|لوحة مفاتيح|ماوس|جهاز)\b'
    ],
    'access': [
        r'\b(password|login|access|authenticate|credentials|account|logon|signin|reset|forgot|locked|unlock|permission|authorization|security)\b',
        r'\b(mot de passe|connexion|compte|réinitialiser|verrouillé|autorisation)\b',
        r'\b(كلمة المرور|تسجيل الدخول|حساب|إعادة تعيين|مقفل|صلاحية)\b'
    ],
    'hr_support': [
        r'\b(billing|invoice|facturation|payment|bill|hr|human|resources|salary|leave|vacation|holiday|employment|recruitment|payroll|benefits|insurance|training|onboarding|offboarding|contract)\b',
        r'\b(facture|paiement|salaire|congé|vacances|emploi|recrutement|contrat)\b',
        r'\b(فاتورة|دفع|راتب|إجازة|توظيف|عقد)\b'
    ],
    'storage': [
        r'\b(storage|space|disk|drive|capacity|memory|quota|cloud|file|folder|directory|upload|download|backup|restore)\b',
        r'\b(stockage|espace|disque|capacité|mémoire|fichier|dossier|sauvegarde)\b',
        r'\b(تخزين|مساحة|قرص|سعة|ملف|نسخ احتياطي)\b'
    ],
    'purchase': [
        r'\b(purchase|buy|procurement|acquisition|software|license|order|vendor|supplier|quote|pricing|cost|budget|approval)\b',
        r'\b(achat|commande|logiciel|licence|fournisseur|devis|coût|budget)\b',
        r'\b(شراء|طلب|برنامج|ترخيص|مورد|سعر|ميزانية)\b'
    ],
    'admin_rights': [
        r'\b(administrative|rights|permission|admin|privileges|authorization|access control|security group|policy|compliance)\b',
        r'\b(administratif|droits|autorisation|privilèges|groupe de sécurité|politique)\b',
        r'\b(إداري|صلاحيات|ترخيص|امتيازات|مجموعة أمان|سياسة)\b'
    ],
    'internal_project': [
        r'\b(project|internal|initiative|task|milestone|deadline|timeline|development|implementation|deployment|testing)\b',
        r'\b(projet|interne|tâche|échéance|développement|mise en œuvre|déploiement|test)\b',
        r'\b(مشروع|داخلي|مهمة|موعد نهائي|تطوير|نشر|اختبار)\b'
    ],
    'miscellaneous': [
        r'\b(network|email|internet|connection|wifi|vpn|outlook|server|website|portal|application|system|service|support)\b',
        r'\b(réseau|courriel|internet|connexion|serveur|site web|portail|application|système)\b',
        r'\b(شبكة|بريد إلكتروني|إنترنت|اتصال|خادم|موقع|بوابة|تطبيق|نظام)\b'
    ]
}

def enhanced_clean_text(text):
    if pd.isna(text) or text == '':
        return ""
    
    text = str(text).lower().strip()
    
    if not text:
        return ""
    
    # cleaning
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\w\s@#]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Language detection function
    def detect_language(word):
        if re.search(r'[؀-ۿ]', word):
            return 'arabic'
        elif re.search(r'[éèêëàâäçîïôöùûüÿ]', word):
            return 'french'
        else:
            return 'english'
    
    
    domain_keywords = []
    for domain, patterns in domain_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text)
            domain_keywords.extend(matches)
    
    words = text.split()
    processed_words = []
    
    for word in words:
        if len(word) < 2:
            continue
            
        if word in domain_keywords:
            processed_words.append(word)
            continue
            
        if word in all_stops:
            continue
            
        lang = detect_language(word)
        
        try:
            if lang == 'english':
                stemmer = SnowballStemmer('english')
                processed_word = stemmer.stem(word)
            elif lang == 'french':
                stemmer = SnowballStemmer('french')
                processed_word = stemmer.stem(word)
            else:
                processed_word = word
                
            if len(processed_word) >= 2:
                processed_words.append(processed_word)
        except:
            processed_words.append(word)
    
    final_words = list(set(processed_words + domain_keywords))
    result = ' '.join(final_words)
    return result if result else "unknown"

print("Applying enhanced text cleaning")
tqdm.pandas()
df['cleaned_document'] = df['Document'].progress_apply(enhanced_clean_text)

print("\nEnhanced Cleaning Examples:")
print("=" * 80)
sample_df = df[['Document', 'cleaned_document']].sample(4)
for idx, row in sample_df.iterrows():
    print(f"\nORIGINAL ({len(row['Document'])} chars):")
    print(f"  {row['Document'][:120]}...")
    print(f"CLEANED ({len(row['cleaned_document'])} chars):")
    print(f"  {row['cleaned_document']}")
    print("-" * 60)

original_avg_len = df['Document'].str.len().mean()
cleaned_avg_len = df['cleaned_document'].str.len().mean()
reduction_pct = ((original_avg_len - cleaned_avg_len) / original_avg_len) * 100

print(f"\nEnhanced Cleaning Effectiveness:")
print(f"Original avg length: {original_avg_len:.0f} chars")
print(f"Cleaned avg length: {cleaned_avg_len:.0f} chars")
print(f"Reduction: {reduction_pct:.1f}%")


#5 dataset split
labels = df['Topic_group'].unique()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print("Enhanced Label mappings:")
for label, idx in label2id.items():
    print(f"  {idx}: {label}")

df['label'] = df['Topic_group'].map(label2id)

X = df['cleaned_document']
y = df['label']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, 
    test_size=0.15,
    random_state=42, 
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15,
    random_state=42,
    stratify=y_temp
)

print(f"\nEnhanced Dataset splits:")
print(f"Training: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print("\nEnhanced datasets created successfully!")


#6 model training function
model_name = "distilbert-base-multilingual-cased" # BJEH RABI AHKI M3A PROFEK 3AL MODEL BERT WLA XLM-ROBERTA 5IR

print(f"Loading enhanced tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def enhanced_tokenize_function(examples):
    tokenized = tokenizer(
        examples['text'],
        padding=False,
        truncation=True,
        max_length=256,
        return_overflowing_tokens=False
    )
    return tokenized

print("Enhanced tokenizing datasets...")
train_dataset = train_dataset.map(enhanced_tokenize_function, batched=True, batch_size=1000)
val_dataset = val_dataset.map(enhanced_tokenize_function, batched=True, batch_size=1000)
test_dataset = test_dataset.map(enhanced_tokenize_function, batched=True, batch_size=1000)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print("Enhanced tokenization completed!")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")



class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Enhanced Class weights:")
for idx, weight in enumerate(class_weights):
    print(f"  {id2label[idx]}: {weight:.2f}")

class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        device = logits.device
        weights = class_weights.to(device)
        
        # Focal Loss implementation
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (0.5 * (1-pt)**2 * ce_loss).mean()
        
        return (focal_loss, outputs) if return_outputs else focal_loss

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

def compute_enhanced_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    true_labels = p.label_ids
    
    acc = accuracy_score(true_labels, preds)
    f1_weighted = f1_score(true_labels, preds, average='weighted')
    f1_macro = f1_score(true_labels, preds, average='macro')
    f1_per_class = f1_score(true_labels, preds, average=None)
    
    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "f1_min": np.min(f1_per_class),
        "f1_max": np.max(f1_per_class)
    }

training_args = TrainingArguments(
    output_dir="./enhanced_results",
    num_train_epochs=12,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    logging_dir="./enhanced_logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=1,
    dataloader_pin_memory=False,
    report_to="none",
    save_total_limit=2,
    max_grad_norm=1.0,
    logging_strategy="steps",
)

print("model training completed")



#7 training model
from transformers import TrainerCallback
class EnhancedTrainingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None:
            print(f"Enhanced Epoch {state.epoch} completed")
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Enhanced Evaluation - Accuracy: {metrics.get('eval_accuracy', 0):.4f}, F1: {metrics.get('eval_f1_weighted', 0):.4f}")

trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_enhanced_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
        EnhancedTrainingCallback()
    ]
)

print("Starting enhanced training")
print("=" * 60)

train_history = trainer.train()

print("\n" + "=" * 60)
print("ENHANCED TRAINING COMPLETED!")
print("=" * 60)

eval_results = trainer.evaluate(test_dataset)
print(f"Enhanced Test Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Enhanced Test F1 (Weighted): {eval_results['eval_f1_weighted']:.4f}")
print(f"Enhanced Test F1 (Macro): {eval_results['eval_f1_macro']:.4f}")
print(f"Worst Class F1: {eval_results['eval_f1_min']:.4f}")
print(f"Best Class F1: {eval_results['eval_f1_max']:.4f}")

test_predictions = trainer.predict(test_dataset)
y_pred = test_predictions.predictions.argmax(-1)
y_true = test_predictions.label_ids

print("\nEnhanced Classification Report:")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(labels))]))

plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=[id2label[i] for i in range(len(labels))],
            yticklabels=[id2label[i] for i in range(len(labels))],
            cbar_kws={'label': 'Number of Samples'})
plt.title('Enhanced Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# 8 save model
import json
trainer.save_model('./enhanced_multilingual_model')
tokenizer.save_pretrained('./enhanced_multilingual_model')

with open('./enhanced_multilingual_model/label_mappings.json', 'w') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f)

print("Enhanced model saved successfully!")



#9 testing for the memes 
model = AutoModelForSequenceClassification.from_pretrained('./enhanced_multilingual_model')
tokenizer = AutoTokenizer.from_pretrained('./enhanced_multilingual_model')

comprehensive_test_cases = [
    # Hardware - English
    ("My laptop won't turn on and the screen is completely black", "Hardware"),
    ("Computer keeps freezing and restarting randomly every hour", "Hardware"),
    ("Printer is not working and showing error code E-05", "Hardware"),
    ("Monitor display is flickering and colors are distorted", "Hardware"),
    ("Keyboard keys are not responding properly", "Hardware"),
    
    # Hardware - French
    ("Mon ordinateur portable ne s'allume pas du tout", "Hardware"),
    ("L'imprimante affiche une erreur et ne fonctionne pas", "Hardware"),
    ("L'écran de mon ordinateur clignote anormalement", "Hardware"),
    ("Le clavier ne répond pas correctement", "Hardware"),
    
    # Hardware - Arabic
    ("الكمبيوتر لا يعمل والشاشة سوداء تماما", "Hardware"),
    ("الطابعة تظهر خطأ ولا تطبع أي شيء", "Hardware"),
    ("الشاشة تومض والألوان غير طبيعية", "Hardware"),
    ("لوحة المفاتيح لا تستجيب بشكل صحيح", "Hardware"),
    
    # Access - English
    ("I forgot my password and cannot login to my email account", "Access"),
    ("Need to reset my credentials for system access", "Access"),
    ("Account locked due to multiple failed login attempts", "Access"),
    ("Request permission to access the shared drive", "Access"),
    ("Cannot authenticate with my current credentials", "Access"),
    
    # Access - French
    ("J'ai oublié mon mot de passe pour me connecter", "Access"),
    ("Besoin de réinitialiser mes identifiants système", "Access"),
    ("Mon compte est verrouillé après plusieurs tentatives", "Access"),
    ("Demande d'autorisation d'accès au dossier partagé", "Access"),
    
    # Access - Arabic
    ("نسيت كلمة المرور ولا يمكنني الدخول للحساب", "Access"),
    ("أحتاج إعادة تعيين بيانات الاعتماد للنظام", "Access"),
    ("حسابي مغلق بسبب محاولات تسجيل دخول خاطئة", "Access"),
    ("طلب صلاحية للوصول إلى المجلد المشترك", "Access"),
    
    # HR Support - English
    ("Request for vacation leave approval for next month", "HR Support"),
    ("Need information about maternity leave policies and benefits", "HR Support"),
    ("Issue with my salary payment for this month", "HR Support"),
    ("Request for training program enrollment", "HR Support"),
    ("Need to update my insurance beneficiary information", "HR Support"),
    
    # HR Support - French
    ("Demande d'approbation de congé pour le mois prochain", "HR Support"),
    ("Information sur les politiques de congé maternité", "HR Support"),
    ("Problème avec le paiement de mon salaire ce mois", "HR Support"),
    ("Demande d'inscription au programme de formation", "HR Support"),
    
    # HR Support - Arabic
    ("طلب موافقة على إجازة للشهر القادم", "HR Support"),
    ("معلومات عن سياسات إجازة الأمومة والمزايا", "HR Support"),
    ("مشكلة في دفع الراتب لهذا الشهر", "HR Support"),
    ("طلب التسجيل في برنامج التدريب", "HR Support"),
    
    # Storage - English
    ("Running out of disk space on my C: drive", "Storage"),
    ("Request for additional cloud storage capacity", "Storage"),
    ("Need to restore files from backup from last week", "Storage"),
    ("Cannot access shared network drive", "Storage"),
    ("Request to increase my storage quota", "Storage"),
    
    # Purchase - English
    ("Need to purchase new software license for Adobe", "Purchase"),
    ("Request approval for buying new office equipment", "Purchase"),
    ("Submit purchase order for new laptops", "Purchase"),
    ("Need vendor quote for annual software renewal", "Purchase"),
    
    # Administrative Rights - English
    ("Request administrative rights to install software", "Administrative rights"),
    ("Need elevated permissions for system configuration", "Administrative rights"),
    ("Request admin access to modify user settings", "Administrative rights"),
    
    # Internal Project - English
    ("Update on internal project timeline and milestones", "Internal Project"),
    ("Request resources for new internal development project", "Internal Project"),
    ("Project deployment scheduled for next week", "Internal Project"),
    
    # Miscellaneous - English
    ("Network connection issues in building A", "Miscellaneous"),
    ("Email server downtime affecting productivity", "Miscellaneous"),
    ("VPN connection problems when working remotely", "Miscellaneous"),
]

print("COMPREHENSIVE ENHANCED MODEL TESTING:")
print("=" * 80)

model.eval()
correct = 0
total = len(comprehensive_test_cases)
results = []

for i, (text, expected) in enumerate(comprehensive_test_cases, 1):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = predictions.argmax(-1).item()
        confidence = predictions.max(-1).values.item()
    
    predicted_label = id2label[predicted_class]
    is_correct = predicted_label == expected
    
    if is_correct:
        correct += 1
    
    status = "✓" if is_correct else "✗"
    results.append({
        'text': text,
        'expected': expected,
        'predicted': predicted_label,
        'correct': is_correct,
        'confidence': confidence
    })
    
    print(f"{status} Test {i:2d}: Expected: {expected:25} Predicted: {predicted_label:25} Confidence: {confidence:.1%}")

print(f"\nCOMPREHENSIVE TEST ACCURACY: {correct/total:.1%} ({correct}/{total})")

# Detailed analysis by category
print("\n" + "="*80)
print("DETAILED CATEGORY ANALYSIS:")
print("="*80)

category_results = {}
for result in results:
    category = result['expected']
    if category not in category_results:
        category_results[category] = {'total': 0, 'correct': 0}
    category_results[category]['total'] += 1
    if result['correct']:
        category_results[category]['correct'] += 1

for category, stats in category_results.items():
    accuracy = stats['correct'] / stats['total']
    print(f"{category:25}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")

# Show errors
print("\n" + "="*80)
print("ERROR ANALYSIS:")
print("="*80)
errors = [r for r in results if not r['correct']]
for error in errors:
    print(f"Expected: {error['expected']:25} → Predicted: {error['predicted']:25}")
    print(f"Text: {error['text'][:80]}...")
    print(f"Confidence: {error['confidence']:.1%}")
    print()

#60% accuracy BADEL MODEL FOR GOD SAKE     