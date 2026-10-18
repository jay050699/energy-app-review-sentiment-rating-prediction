"""Consolidate the models compared on validation data by src/train.py.

This script does not select models using the held-out test partition and does
not retrain or overwrite any saved model. The final model selection and test
scoring belong to src/train.py.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def main():
    tables = []
    for task, filename in [
        ('sentiment','final_baseline_vs_tuned_model_comparison.csv'),
        ('rating','final_rating_baseline_vs_tuned_comparison.csv'),
    ]:
        frame = pd.read_csv(REPORTS / filename)
        frame.insert(0,'evaluation_set','validation')
        frame.insert(0,'task',task)
        tables.append(frame)
    combined = pd.concat(tables,ignore_index=True)
    output = REPORTS / 'model_improvement_experiments.csv'
    if not output.is_file():
        raise FileNotFoundError(f'Missing original report path: {output}')
    combined.to_csv(output,index=False)
    print('Saved validation experiment summary:', output)
    print(combined[['task','model_name','accuracy','macro_f1']].to_string(index=False))


if __name__=='__main__':
    main()
