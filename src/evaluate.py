import argparse
import json
import os
import logging
from src.utils import set_seed, setup_logging, get_device
import config
from src.data_loader import load_csv
from src.preprocess import preprocess_texts, build_prompt
from src.features import TFIDFExtractor
from src.model import SklearnModel, TransformerModel
from scoring import evaluate_predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on dev set.")
    parser.add_argument('--data', required=True, help="Path to dev CSV")
    parser.add_argument('--artifacts', required=True, help="Artifacts dir")
    args = parser.parse_args()
    
    setup_logging()
    set_seed(config.SEED)
    logging.info(f"Device: {get_device()}")
    
    with open(os.path.join(args.artifacts, 'config.json'), 'r') as f:
        art_config = json.load(f)
    with open(os.path.join(args.artifacts, 'label_map.json'), 'r') as f:
        label_map = json.load(f)
    id2label = label_map['id2label']
    
    df = load_csv(args.data, require_target=False)
    test_lists = df.get('test_list', [None]*len(df)).tolist()
    prompts = [build_prompt(p, t) for p, t in zip(df['prompt'], test_lists)]
    prompts = preprocess_texts(prompts)
    
    if 'target' not in df.columns:
        logging.warning("No targets in data; skipping metrics")
        return
    
    targets = df['target'].tolist()
    label2id = label_map['label2id']
    true_labels = [label2id.get(t, -1) for t in targets]
    if -1 in true_labels:
        logging.warning("Unknown labels in data; skipping metrics")
        return
    
    if art_config['backend'] == 'sklearn':
        extractor = TFIDFExtractor({})  # Params not needed for load
        extractor.load(os.path.join(args.artifacts, 'vectorizer.pkl'))
        model = SklearnModel({})
        model.load(os.path.join(args.artifacts, 'model.pkl'))
        X = extractor.transform(prompts)
        pred_ids = model.predict(X)
    elif art_config['backend'] == 'transformer':
        if art_config['is_fine_tune']:
            model = TransformerModel(config.MODEL_NAME, config.MAX_LENGTH, is_fine_tune=True, num_labels=len(id2label), id2label=id2label, label2id=label2id)
            # Load saved
            model.model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.artifacts, 'model'))
            model.tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.artifacts, 'tokenizer'))
            model.model.to(get_device())
        else:
            model = TransformerModel(config.MODEL_NAME, config.MAX_LENGTH, is_fine_tune=False)
            model.model.load_state_dict(torch.load(os.path.join(args.artifacts, 'encoder.pt')))
            model.tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.artifacts, 'tokenizer'))
            model.model.to(get_device())
            model.classifier = joblib.load(os.path.join(args.artifacts, 'classifier.pkl'))
        pred_ids = model.predict(prompts)
    
    metrics = evaluate_predictions(true_labels, pred_ids, test_lists)
    logging.info(f"Metrics: {metrics}")
    with open(os.path.join(args.artifacts, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()