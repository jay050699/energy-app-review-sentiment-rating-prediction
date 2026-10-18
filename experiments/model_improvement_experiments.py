"""Targeted, validation-only experiments for existing Energy App Review NLP project.

Run from project root after training the existing final models:
    python -u experiments/model_improvement_experiments.py

The test set is NEVER used to compare or select candidates. This script updates
ONLY reports/model_improvement_experiments.csv (an already tracked report) and
never touches saved models or final_model_summary.csv.
"""
from __future__ import annotations
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.naive_bayes import ComplementNB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import load_dataset, make_partitions, sentiment_candidates, rating_candidates

OUTPUT = ROOT / 'reports/model_improvement_experiments.csv'


def variants(task):
    if task == 'sentiment':
        base = sentiment_candidates()['TF-IDF + LinearSVC']
        variants = [
            ('Current selected: word TF-IDF + LinearSVC', clone(base)),
            ('LinearSVC C=0.35', clone(base).set_params(model__C=0.35)),
            ('LinearSVC C=2.0', clone(base).set_params(model__C=2.0)),
            ('LinearSVC neutral weight 1.5x', clone(base).set_params(
                model__class_weight={'Negative':1, 'Neutral':9.5, 'Positive':1})),
            ('LinearSVC neutral weight 2x', clone(base).set_params(
                model__class_weight={'Negative':1, 'Neutral':13, 'Positive':1})),
            ('TF-IDF up to 20K features', clone(base).set_params(
                vectorizer__max_features=20000)),
        ]
    else:
        base = rating_candidates()['CountVectorizer + Naive Bayes']
        variants = [
            ('Current selected: CountVectorizer + Naive Bayes', clone(base)),
            ('Naive Bayes alpha=0.25', clone(base).set_params(model__alpha=0.25)),
            ('Naive Bayes alpha=0.5', clone(base).set_params(model__alpha=0.5)),
            ('Naive Bayes alpha=2.0', clone(base).set_params(model__alpha=2.0)),
            ('Naive Bayes alpha=5.0', clone(base).set_params(model__alpha=5.0)),
            ('CountVectorizer 15K features', clone(base).set_params(
                vectorizer__max_features=15000, vectorizer__min_df=2)),
            ('ComplementNB alpha=1.0', clone(base).set_params(
                model=ComplementNB(alpha=1.0))),
        ]
    return variants


def evaluate(task, y, pred):
    y=np.asarray(y)
    pred=np.asarray(pred)
    labels=['Negative','Neutral','Positive'] if task=='sentiment' else [1,2,3,4,5]
    per_class=f1_score(y,pred,labels=labels,average=None,zero_division=0)
    result={
       'accuracy':round(float(accuracy_score(y,pred)),4),
       'macro_f1':round(float(f1_score(y,pred,average='macro',zero_division=0)),4),
       'weighted_f1':round(float(f1_score(y,pred,average='weighted',zero_division=0)),4),
       'mae':round(float(mean_absolute_error(y,pred)),4) if task=='rating' else '',
    }
    for label,value in zip(labels,per_class):
        result[f'f1_{str(label).lower()}']=round(float(value),4)
    return result


def main():
    if not OUTPUT.is_file():
        raise FileNotFoundError(f'Expected existing report missing: {OUTPUT}')
    df,groups=load_dataset()
    train_idx,val_idx,_test_idx=make_partitions(df,groups)
    X_train=df['clean_review_text'].iloc[train_idx]
    X_val=df['clean_review_text'].iloc[val_idx]
    rows=[]
    for task in ['sentiment','rating']:
        target='sentiment' if task=='sentiment' else 'rating'
        y_train=df[target].iloc[train_idx]
        y_val=df[target].iloc[val_idx]
        print(f'\n=== {task.upper()} VALIDATION-ONLY EXPERIMENTS ===',flush=True)
        baseline=None
        for index,(name,pipe) in enumerate(variants(task)):
            t=time.perf_counter()
            pipe.fit(X_train,y_train)
            pred=pipe.predict(X_val)
            res=evaluate(task,y_val,pred)
            if index==0:baseline=res['macro_f1']
            row={'task':task,'evaluation_set':'validation','candidate':name,
                 'validation_examples':len(val_idx),
                 'macro_f1_gain_vs_current':round(res['macro_f1']-baseline,4),
                 'time_seconds':round(time.perf_counter()-t,1),**res}
            rows.append(row)
            focus=res.get('f1_neutral','') if task=='sentiment' else f"2={res['f1_2']} 3={res['f1_3']} 4={res['f1_4']}"
            print(f"{name:<48} Macro F1={res['macro_f1']:.4f}  Accuracy={res['accuracy']:.4f}  Focus F1={focus}",flush=True)
    out=pd.DataFrame(rows)
    out.to_csv(OUTPUT,index=False)
    print('\n=== BEST BY VALIDATION MACRO F1 (NOT TEST RESULTS) ===',flush=True)
    for task,g in out.groupby('task'):
        best=g.sort_values(['macro_f1','accuracy'],ascending=False).iloc[0]
        print(f"{task}: {best.candidate} | Macro F1={best.macro_f1} | gain={best.macro_f1_gain_vs_current:+.4f}",flush=True)
    print('\nUpdated existing report:',OUTPUT,flush=True)
    print('Saved models and final held-out test results were NOT modified.',flush=True)

if __name__=='__main__':
    main()
