import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

train_path = os.path.join(os.getcwd(), 'train.csv')
test_path = os.path.join(os.getcwd(), 'test.csv')
sample_path = os.path.join(os.getcwd(), 'sample_submission.csv')


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub_view = pd.read_csv(sample_path)


train['dept'] = train['label'].apply(lambda x: x.split('|')[0])
train['urgency'] = train['label'].apply(lambda x: x.split('|')[1])

train['text_len'] = train['body'].apply(len)
train['word_count'] = train['body'].apply(lambda x: len(str(x).split()))

import re

class GrievanceTextProcessor:
    """Preprocesses code-mixed grievance text."""

    FILLER_WORDS = {
        'sir', 'namaskara', 'namaste', 'pranam', 'ji', 'saheb', 'madam',
        'dear', 'team', 'please', 'kindly', 'request', 'thank', 'thanks',
        'urgent', 'immediately', 'asap', 'soon', 'quickly',
    }

    CATEGORY_KEYWORDS = {
        'roads_transport': ['road', 'rasta', 'street', 'highway', 'bridge', 'pothole', 'traffic', 
                            'bus', 'auto', 'vehicle', 'accident', 'signal', 'footpath', 'drain'],
        'water_supply': ['water', 'paani', 'neeru', 'tap', 'pipeline', 'borewell', 'tanker', 
                         'drinking', 'supply', 'shortage', 'leakage', 'pollution'],
        'electricity': ['power', 'current', 'bijli', 'vijjut', 'transformer', 'wire', 'pole',
                        'outage', 'bill', 'meter', 'connection', 'generator', 'solar'],
        'healthcare': ['hospital', 'aspatre', 'doctor', 'daktar', 'medicine', 'ambulance',
                       'patient', 'fever', 'disease', 'clinic', 'health', 'treatment'],
        'education': ['school', 'shale', 'college', 'student', 'teacher', 'master',
                      'exam', 'result', 'admission', 'fee', 'books', 'midday meal'],
        'welfare_schemes': ['ration', 'pension', 'scheme', 'yojane', 'subsidy', 'card',
                            'aadhar', 'benefit', 'eligibility', 'application', 'approval'],
        'law_and_order': ['police', 'theft', 'chain snatching', 'robbery', 'crime', 'case',
                          'complaint', 'fir', 'court', 'lawyer', 'violence', 'safety'],
        'agriculture_irrigation': ['farm', 'kheti', 'crop', 'water canal', 'irrigation', 
                                   'fertilizer', 'seed', 'pest', 'land', 'harvest', 'mandi']
    }

    URGENCY_KEYWORDS = {
        'critical': ['death', 'dying', 'emergency', 'accident', 'fire', 'collapse', 
                     'flood', 'danger', 'injured', 'unconscious', 'poisoning'],
        'high': ['no supply', 'broken', 'stopped', 'three days', 'one week', 'children suffering',
                 'elderly', 'pregnant', 'school closed', 'hospital closed'],
        'routine': ['application', 'pending', 'request', 'suggestion', 'improve', 'quality',
                    'cleaning', 'maintenance', 'information', 'clarification']
    }

    def __init__(self):
        self.patterns = {
            'repeat_chars': re.compile(r'(.)\1{2,}'),
            'laughter': re.compile(r'\b(haha|hehe|lol|lmao|rofl)\b'),
            'mentions': re.compile(r'@\w+'),
            'urls': re.compile(r'http\S+|www\.\S+'),
            'phone': re.compile(r'\b\d{10}\b'),
            'extra_spaces': re.compile(r'\s+'),
        }

    def normalize(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower().strip()
        text = self.patterns['repeat_chars'].sub(r'\1\1', text)
        text = self.patterns['urls'].sub('', text)
        text = self.patterns['mentions'].sub('', text)
        text = self.patterns['phone'].sub('<PHONE>', text)
        text = self.patterns['laughter'].sub('<LAUGH>', text)
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!,\-]', ' ', text)
        text = self.patterns['extra_spaces'].sub(' ', text)
        return text.strip()

    def extract_features(self, text):
        words = text.split()
        clean_words = [w for w in words if w not in self.FILLER_WORDS]
        has_devanagari = any('\u0900' <= c <= '\u097f' for c in text)

        return {
            'word_count': len(words),
            'clean_word_count': len(clean_words),
            'has_hindi': int(has_devanagari),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': round(sum(1 for c in text if c.isupper()) / max(len(text), 1), 3),
            'time_refs': len(re.findall(r'\b(daily|weekly|monthly|today|yesterday|tomorrow|\d+\s*(day|week|month|hour))\b', text)),
            'numeric_refs': len(re.findall(r'\b\d+\+?\s*(hour|day|week|month|year|km|meter|rupee|rs)\b', text)),
            'negation_count': len(re.findall(r'\b(no|not|never|without|missing|lack|absent)\b', text)),
        }

    def keyword_score(self, text):
        text_lower = text.lower()
        scores = {}
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            scores[f'kw_{cat}'] = sum(1 for kw in keywords if kw in text_lower)
        for urg, keywords in self.URGENCY_KEYWORDS.items():
            scores[f'kw_{urg}'] = sum(1 for kw in keywords if kw in text_lower)
        return scores

    def process(self, text):
        clean = self.normalize(text)
        features = self.extract_features(clean)
        kw_scores = self.keyword_score(clean)
        return clean, features, kw_scores

# Process all texts
processor = GrievanceTextProcessor()

print("Processing train texts...")
train_clean, train_features, train_kw = [], [], []
for text in tqdm(train['body']):
    c, f, k = processor.process(text)
    train_clean.append(c)
    train_features.append(f)
    train_kw.append(k)

print("Processing test texts...")
test_clean, test_features, test_kw = [], [], []
for text in tqdm(test['body']):
    c, f, k = processor.process(text)
    test_clean.append(c)
    test_features.append(f)
    test_kw.append(k)

train['clean_body'] = train_clean
test['clean_body'] = test_clean

print("\nOriginal:")
print(train['body'].iloc[0][:200])
print("\nCleaned:")
print(train['clean_body'].iloc[0][:200])

train['dept_label'] = train['dept']
train['urgency_label'] = train['urgency']

dept_encoder = LabelEncoder()
urgency_encoder = LabelEncoder()

train['dept_encoded'] = dept_encoder.fit_transform(train['dept_label'])
train['urgency_encoded'] = urgency_encoder.fit_transform(train['urgency_label'])

DEPARTMENTS = dept_encoder.classes_.tolist()
URGENCY_LEVELS = urgency_encoder.classes_.tolist()


class GrievanceClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_departments=8, num_urgency=3, dropout=0.3):
        super().__init__()
        self.config = AutoModel.from_pretrained(model_name).config
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size

        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout / 2),
        )

        self.dept_classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 4, num_departments)
        )

        self.urgency_classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 4, num_urgency)
        )

    def forward(self, input_ids, attention_mask, dept_labels=None, urgency_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        shared_repr = self.shared(pooled)

        dept_logits = self.dept_classifier(shared_repr)
        urgency_logits = self.urgency_classifier(shared_repr)

        loss = None
        if dept_labels is not None and urgency_labels is not None:
            dept_loss = nn.CrossEntropyLoss()(dept_logits, dept_labels)
            urgency_loss = nn.CrossEntropyLoss()(urgency_logits, urgency_labels)
            loss = dept_loss + 0.4 * urgency_loss

        return {
            'loss': loss,
            'dept_logits': dept_logits,
            'urgency_logits': urgency_logits
        }

class GrievanceDataset(torch.utils.data.Dataset):
    def __init__(self, texts, dept_labels, urgency_labels, tokenizer, max_len=256):
        self.texts = texts
        self.dept_labels = dept_labels
        self.urgency_labels = urgency_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'dept_label': torch.tensor(self.dept_labels[idx], dtype=torch.long),
            'urgency_label': torch.tensor(self.urgency_labels[idx], dtype=torch.long)
        }


def compute_class_weights(labels):
    counts = np.bincount(labels)
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float)

def evaluate_model(model, dataloader, device):
    model.eval()
    dept_preds, dept_true = [], []
    urgency_preds, urgency_true = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)

            dept_preds.extend(torch.argmax(outputs['dept_logits'], dim=1).cpu().numpy())
            urgency_preds.extend(torch.argmax(outputs['urgency_logits'], dim=1).cpu().numpy())
            dept_true.extend(batch['dept_label'].numpy())
            urgency_true.extend(batch['urgency_label'].numpy())

    dept_f1 = f1_score(dept_true, dept_preds, average='weighted')
    urgency_f1 = f1_score(urgency_true, urgency_preds, average='weighted')
    combined_f1 = (dept_f1 + urgency_f1) / 2

    dept_acc = accuracy_score(dept_true, dept_preds)
    urgency_acc = accuracy_score(urgency_true, urgency_preds)

    return {
        'dept_f1': dept_f1, 'urgency_f1': urgency_f1, 'combined_f1': combined_f1,
        'dept_acc': dept_acc, 'urgency_acc': urgency_acc,
        'dept_preds': dept_preds, 'urgency_preds': urgency_preds,
        'dept_true': dept_true, 'urgency_true': urgency_true
    }

MODEL_NAME = "xlm-roberta-base"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 256
N_SPLITS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
texts = train['clean_body'].tolist()
dept_labels = train['dept_encoded'].values
urgency_labels = train['urgency_encoded'].values

# Stratified K-Fold on department (more stable than 24 classes)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_results = []
model_paths = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, dept_labels)):
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/{N_SPLITS}")
    print(f"{'='*60}")

    tr_texts = [texts[i] for i in train_idx]
    tr_depts = dept_labels[train_idx]
    tr_urgency = urgency_labels[train_idx]

    val_texts = [texts[i] for i in val_idx]
    val_depts = dept_labels[val_idx]
    val_urgency = urgency_labels[val_idx]

    train_dataset = GrievanceDataset(tr_texts, tr_depts, tr_urgency, tokenizer, MAX_LEN)
    val_dataset = GrievanceDataset(val_texts, val_depts, val_urgency, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

    model = GrievanceClassifier(MODEL_NAME).to(device)

    dept_weights = compute_class_weights(tr_depts).to(device)
    urgency_weights = compute_class_weights(tr_urgency).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dept_l = batch['dept_label'].to(device)
            urgency_l = batch['urgency_label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, dept_l, urgency_l)

            dept_loss = nn.CrossEntropyLoss(weight=dept_weights)(outputs['dept_logits'], dept_l)
            urgency_loss = nn.CrossEntropyLoss(weight=urgency_weights)(outputs['urgency_logits'], urgency_l)
            loss = dept_loss + 0.4 * urgency_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"Val F1={val_metrics['combined_f1']:.4f} "
              f"(D={val_metrics['dept_f1']:.4f}, U={val_metrics['urgency_f1']:.4f})")

        if val_metrics['combined_f1'] > best_f1:
            best_f1 = val_metrics['combined_f1']
            model_path = f'best_model_fold_{fold}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'fold': fold
            }, model_path)
            print(f"  -> Saved new best model (F1={best_f1:.4f})")

    fold_results.append(best_f1)
    model_paths.append(f'best_model_fold_{fold}.pt')
    print(f"Fold {fold+1} Best F1: {best_f1:.4f}")

print(f"\n{'='*60}")
print("Cross-Validation Summary")
print(f"{'='*60}")
print(f"Mean F1: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
print(f"All folds: {[f'{f:.4f}' for f in fold_results]}")

best_fold = np.argmax(fold_results)
print(f"Best fold: {best_fold + 1} (F1 = {fold_results[best_fold]:.4f})")

checkpoint = torch.load(model_paths[best_fold])
print("\nValidation metrics:")
for k, v in checkpoint['metrics'].items():
    if not isinstance(v, list):
        print(f"  {k}: {v:.4f}")

# Confusion matrices
dept_preds = checkpoint['metrics']['dept_preds']
dept_true = checkpoint['metrics']['dept_true']
urgency_preds = checkpoint['metrics']['urgency_preds']
urgency_true = checkpoint['metrics']['urgency_true']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# cm_dept = confusion_matrix(dept_true, dept_preds)
# sns.heatmap(cm_dept, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=DEPARTMENTS, yticklabels=DEPARTMENTS, ax=axes[0])
# axes[0].set_title('Department Confusion Matrix')
# axes[0].set_xlabel('Predicted')
# axes[0].set_ylabel('True')

# cm_urg = confusion_matrix(urgency_true, urgency_preds)
# sns.heatmap(cm_urg, annot=True, fmt='d', cmap='Blues',
#             xticklabels=URGENCY_LEVELS, yticklabels=URGENCY_LEVELS, ax=axes[1])
# axes[1].set_title('Urgency Confusion Matrix')
# axes[1].set_xlabel('Predicted')
# axes[1].set_ylabel('True')

# plt.tight_layout()
# plt.savefig('confusion_matrices.png', dpi=150)
# plt.show()


print("Training final model on full training set...")

full_dataset = GrievanceDataset(texts, dept_labels, urgency_labels, tokenizer, MAX_LEN)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

final_model = GrievanceClassifier(MODEL_NAME).to(device)

dept_weights = compute_class_weights(dept_labels).to(device)
urgency_weights = compute_class_weights(urgency_labels).to(device)

optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(full_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

for epoch in range(EPOCHS):
    final_model.train()
    total_loss = 0

    for batch in tqdm(full_loader, desc=f"Final Epoch {epoch+1}/{EPOCHS}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dept_l = batch['dept_label'].to(device)
        urgency_l = batch['urgency_label'].to(device)

        optimizer.zero_grad()
        outputs = final_model(input_ids, attention_mask, dept_l, urgency_l)

        dept_loss = nn.CrossEntropyLoss(weight=dept_weights)(outputs['dept_logits'], dept_l)
        urgency_loss = nn.CrossEntropyLoss(weight=urgency_weights)(outputs['urgency_logits'], urgency_l)
        loss = dept_loss + 0.4 * urgency_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    print(f"Final Epoch {epoch+1}: Loss={total_loss/len(full_loader):.4f}")

torch.save(final_model.state_dict(), 'final_model.pt')
print("Final model saved.")

def predict_ensemble(test_texts, model_paths, tokenizer, device, batch_size=32):
    """Ensemble predictions from all fold models."""

    dummy_depts = [0] * len(test_texts)
    dummy_urgency = [0] * len(test_texts)

    test_dataset = GrievanceDataset(test_texts, dummy_depts, dummy_urgency, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_models = []
    for path in model_paths:
        model = GrievanceClassifier(MODEL_NAME).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        all_models.append(model)

    all_dept_probs = []
    all_urgency_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            batch_dept_probs = []
            batch_urgency_probs = []

            for model in all_models:
                outputs = model(input_ids, attention_mask)
                batch_dept_probs.append(torch.softmax(outputs['dept_logits'], dim=1).cpu().numpy())
                batch_urgency_probs.append(torch.softmax(outputs['urgency_logits'], dim=1).cpu().numpy())

            avg_dept = np.mean(batch_dept_probs, axis=0)
            avg_urgency = np.mean(batch_urgency_probs, axis=0)

            all_dept_probs.append(avg_dept)
            all_urgency_probs.append(avg_urgency)

    dept_probs = np.concatenate(all_dept_probs, axis=0)
    urgency_probs = np.concatenate(all_urgency_probs, axis=0)

    dept_preds = np.argmax(dept_probs, axis=1)
    urgency_preds = np.argmax(urgency_probs, axis=1)

    labels = []
    for d, u in zip(dept_preds, urgency_preds):
        labels.append(f"{DEPARTMENTS[d]}|{URGENCY_LEVELS[u]}")

    return labels, dept_probs, urgency_probs

# Generate predictions
test_texts = test['clean_body'].tolist()
predicted_labels, dept_probs, urgency_probs = predict_ensemble(
    test_texts, model_paths, tokenizer, device
)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'label': predicted_labels
})

submission.to_csv('output_submission.csv', index=False)
print(f"Submission saved! Shape: {submission.shape}")
print("\nLabel distribution:")
print(submission['label'].value_counts().head(15))

# Confidence analysis
confidence_scores = [(d.max() + u.max()) / 2 for d, u in zip(dept_probs, urgency_probs)]
print(f"\nConfidence stats:")
print(f"  Mean: {np.mean(confidence_scores):.3f}")
print(f"  Min: {np.min(confidence_scores):.3f}")
print(f"  Max: {np.max(confidence_scores):.3f}")

# Low confidence predictions
low_conf_idx = np.argsort(confidence_scores)[:10]
print("\nLowest confidence predictions:")
for idx in low_conf_idx:
    print(f"  {submission.iloc[idx]['id']}: {submission.iloc[idx]['label']} (conf={confidence_scores[idx]:.3f})")
    print(f"    Text: {test['body'].iloc[idx][:100]}...")


# ## 9. Error Analysis on Validation

best_checkpoint = torch.load(model_paths[best_fold])
val_dept_preds = best_checkpoint['metrics']['dept_preds']
val_dept_true = best_checkpoint['metrics']['dept_true']
val_urg_preds = best_checkpoint['metrics']['urgency_preds']
val_urg_true = best_checkpoint['metrics']['urgency_true']

# Get validation indices
_, val_idx = list(skf.split(texts, dept_labels))[best_fold]
val_df = train.iloc[val_idx].copy()
val_df['pred_dept'] = [DEPARTMENTS[p] for p in val_dept_preds]
val_df['pred_urgency'] = [URGENCY_LEVELS[p] for p in val_urg_preds]
val_df['pred_label'] = val_df['pred_dept'] + '|' + val_df['pred_urgency']
val_df['correct'] = val_df['label'] == val_df['pred_label']

print(f"Overall accuracy: {val_df['correct'].mean():.4f}")
print(f"\nMisclassified by department:")
print(val_df[val_df['dept'] != val_df['pred_dept']]['dept'].value_counts())

print(f"\nMisclassified by urgency:")
print(val_df[val_df['urgency'] != val_df['pred_urgency']]['urgency'].value_counts())

# Show some failure cases
print(f"\n{'='*60}")
print("Failure Case Examples")
print(f"{'='*60}")
failures = val_df[~val_df['correct']].sample(5, random_state=42)
for _, row in failures.iterrows():
    print(f"\nID: {row['id']}")
    print(f"True: {row['label']} | Pred: {row['pred_label']}")
    print(f"Text: {row['body'][:200]}...")

# Save label encoders and config
import pickle

artifacts = {
    'dept_encoder': dept_encoder,
    'urgency_encoder': urgency_encoder,
    'departments': DEPARTMENTS,
    'urgency_levels': URGENCY_LEVELS,
    'model_name': MODEL_NAME,
    'max_len': MAX_LEN,
    'cv_scores': fold_results
}

with open('artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Artifacts saved.")
print("\nFiles in output:")
for f in os.listdir('.'):
    if f.endswith(('.csv', '.pt', '.pkl', '.png')):
        size = os.path.getsize(f) / 1024
        print(f"  {f}: {size:.1f} KB")
