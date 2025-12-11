import os
import json
from pprint import pprint

def load_summary(output_dir='confusion_and_roc'):
    fp = os.path.join(output_dir, 'results_summary.json')
    if not os.path.exists(fp):
        raise FileNotFoundError('Results summary not found. Run train.py first.')
    with open(fp) as f:
        return json.load(f)

def print_summary(output_dir='confusion_and_roc'):
    summary = load_summary(output_dir)
    print('Model comparison:')
    for name, info in summary.items():
        print(f"- {name}: Accuracy={info.get('accuracy'):.4f}, ROC_AUC={info.get('roc_auc')}")
    print('\nDetailed files saved under', output_dir)

if __name__ == '__main__':
    print_summary()
