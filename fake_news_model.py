# Fake News Detection — extracted from supplied notebook
# Install dependencies before running.


# # Fake News Detection (improved)
# Run the cells top to bottom. Sections marked **OPTIONAL** can be skipped.

# ## 1. Setup

# %pip install -q pandas numpy scikit-learn joblib matplotlib

import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score

RANDOM_STATE = 42

# ## 2. Load data, label, remove empty and duplicate rows

df_fake = pd.read_csv('Fake.csv'); df_fake['label'] = 0   # 0 = Fake
df_true = pd.read_csv('True.csv'); df_true['label'] = 1   # 1 = Real

df = pd.concat([df_fake, df_true], ignore_index=True)
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df['date'] = pd.to_datetime(df['date'], errors='coerce')   # used only in the optional time-split section

# Drop empty and duplicate articles BEFORE splitting (prevents train/test leakage)
before = len(df)
df = df[df['content'].str.strip().str.len() > 20]
df = df.drop_duplicates(subset='content').reset_index(drop=True)
print(f"Removed {before - len(df)} empty/duplicate rows -> {len(df)} remain")
print(df['label'].value_counts())

# ## 3. Clean text and split

def clean_text(text):
    text = str(text)
    text = re.sub(r'^.{0,120}?\(reuters\)\s*-?\s*', '', text, flags=re.I)  # "WASHINGTON (Reuters) -"
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)   # URLs
    text = re.sub(r'<.*?>', ' ', text)                    # HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text)                 # punctuation and digits -> space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['content'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'],
    test_size=0.20, random_state=RANDOM_STATE, stratify=df['label']
)
print(len(X_train), "train /", len(X_test), "test")

# ## 4. Define models (each has its own TF-IDF inside a Pipeline)

stop_words = list(ENGLISH_STOP_WORDS | {'reuters', 'said', 'image', 'featured', 'getty', 'via'})

def make_pipeline(clf):
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                  stop_words=stop_words, min_df=3, max_df=0.9,
                                  sublinear_tf=True)),
        ('clf', clf),
    ])

models = {
    'Logistic Regression': make_pipeline(LogisticRegression(max_iter=1000, C=5)),
    'Linear SVM':          make_pipeline(LinearSVC(C=1.0)),
    'Passive Aggressive':  make_pipeline(PassiveAggressiveClassifier(max_iter=1000, random_state=RANDOM_STATE)),
    'Multinomial NB':      make_pipeline(MultinomialNB()),
}

# ## 5. Compare models with 5-fold CV on the TRAIN set only

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, pipe in models.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=['accuracy', 'f1'], n_jobs=-1)
    results[name] = {
        'CV Accuracy': scores['test_accuracy'].mean(),
        'CV F1': scores['test_f1'].mean(),
        'Std (F1)': scores['test_f1'].std(),
    }

results_df = pd.DataFrame(results).T.sort_values('CV F1', ascending=False)
print(results_df.round(4))

best_name = results_df.index[0]
baseline_cv_f1 = results_df.loc[best_name, 'CV F1']
print(f"\nBest model by CV F1: {best_name} ({baseline_cv_f1:.4f})")

# ## 6. Fit the best model and evaluate ONCE on the test set

best_pipe = models[best_name]
best_pipe.fit(X_train, y_train)
preds = best_pipe.predict(X_test)

print(classification_report(y_test, preds, target_names=['Fake', 'Real'], digits=4))
ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=['Fake', 'Real'], cmap='Blues')
plt.title(f'{best_name} - Test Set')
plt.show()

# ## 7. What did the model learn? (linear models)

tfidf = best_pipe.named_steps['tfidf']
clf = best_pipe.named_steps['clf']

if hasattr(clf, 'coef_'):
    feats = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_.ravel()
    print("Strongest FAKE indicators:", list(feats[np.argsort(coefs)[:20]]))
    print("Strongest REAL indicators:", list(feats[np.argsort(coefs)[-20:]]))

# ## 8. Save the baseline model

joblib.dump(best_pipe, 'fake_news_pipeline_baseline.pkl')
print("Saved baseline model.")

# ## 9. Error analysis (look at the mistakes)

mask = preds != y_test.values
err_df = df.loc[X_test.index[mask], ['title', 'label']].copy()
err_df['predicted'] = preds[mask]
err_df['true'] = err_df['label'].map({0: 'Fake', 1: 'Real'})
err_df['predicted'] = err_df['predicted'].map({0: 'Fake', 1: 'Real'})
print(f"{mask.sum()} errors out of {len(y_test)}")
print(err_df[['title', 'true', 'predicted']].head(15).to_string())

# Are errors linked to article length?
df['word_count'] = df['cleaned'].str.split().str.len()
print("\nMean words, all test articles :", round(df.loc[X_test.index, 'word_count'].mean(), 1))
print("Mean words, misclassified     :", round(df.loc[X_test.index[mask], 'word_count'].mean(), 1))

# ## 10. Decision threshold check (catch more fakes vs. false alarms)

if hasattr(best_pipe, 'decision_function'):
    scores = best_pipe.decision_function(X_test)
    yt = y_test.values
    rows = []
    for t in [-0.5, -0.25, 0, 0.25, 0.5]:
        p = (scores > t).astype(int)
        rows.append({
            'threshold': t,
            'fake recall': round(((p == 0) & (yt == 0)).sum() / (yt == 0).sum(), 4),
            'real recall': round(((p == 1) & (yt == 1)).sum() / (yt == 1).sum(), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print("This model has no decision_function; skip this section.")

# ## 11. Train the final model on ALL data
# This defines `final_pipe`, which section 12 saves for the app.

final_pipe = clone(models['Logistic Regression'])
final_pipe.fit(df['cleaned'], df['label'])
import sklearn; print('scikit-learn version:', sklearn.__version__)  # pin this in requirements.txt

# ## 12. Save the final model and add a prediction helper

joblib.dump(final_pipe, 'fake_news_pipeline.pkl')   # this is the file app.py loads
print("Saved fake_news_pipeline.pkl")

def predict_news(text, pipeline=None):
    pipeline = pipeline or final_pipe
    cleaned = clean_text(text)
    label = pipeline.predict([cleaned])[0]
    clf = pipeline.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        conf = pipeline.predict_proba([cleaned])[0].max()
    else:
        conf = 1 / (1 + np.exp(-abs(pipeline.decision_function([cleaned])[0])))
    return ('Real' if label == 1 else 'Fake'), round(float(conf), 3)

print(predict_news("Scientists announce breakthrough in battery technology at university lab."))

# ## 13. OPTIONAL: time-based split check

df_t = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
cut = int(len(df_t) * 0.8)
Xt_train, Xt_test = df_t['cleaned'][:cut], df_t['cleaned'][cut:]
yt_train, yt_test = df_t['label'][:cut], df_t['label'][cut:]
print("Train up to:", df_t['date'][cut - 1].date(), "| Test from:", df_t['date'][cut].date())

time_pipe = clone(final_pipe)
time_pipe.fit(Xt_train, yt_train)
print(classification_report(yt_test, time_pipe.predict(Xt_test), target_names=['Fake', 'Real'], digits=4))
