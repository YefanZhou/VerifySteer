import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import logging

parser = argparse.ArgumentParser(description='Evaluate MLP probe on extracted hidden states')
parser.add_argument('--val_hidden_states_path_lst', type=str, nargs='+', required=True,
                    help='Path(s) to hidden states .pt file(s)')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the saved probe checkpoint (.pt)')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory to write evaluation results')
parser.add_argument('--dataset_name', type=str, default=None,
                    help='Dataset name appended to output path (e.g. gsm8k). '
                         'Inferred from val path if not provided.')
parser.add_argument('--layer_idx', type=int, default=-1)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if args.dataset_name is None:
    parts = Path(args.val_hidden_states_path_lst[0]).parts
    try:
        hs_idx = parts.index('hidden_states')
        args.dataset_name = parts[hs_idx - 2]
    except (ValueError, IndexError):
        args.dataset_name = 'eval'

output_dir = Path(args.model_path).parent / args.dataset_name
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / f'eval_layer_{args.layer_idx}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logging.info(f"Arguments: {args}")


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        if num_layers == 2:
            layers.extend([
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            ])
        elif num_layers == 3:
            layers.extend([
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ])
        else:
            raise ValueError(f"num_layers must be 2 or 3, got {num_layers}")
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch_states, batch_labels in dataloader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.float().to(device)

            logits = model(batch_states).squeeze()
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': avg_loss, 'accuracy': accuracy,
        'precision': precision, 'recall': recall,
        'f1': f1, 'auc': auc,
        'confusion_matrix': cm.tolist()
    }, all_probs, all_preds, all_labels


def get_predictions(model, dataloader, device, sample_ids):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch_states, batch_labels in dataloader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.float().to(device)

            logits = model(batch_states).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch_labels.cpu().numpy().tolist())

    return [
        {'id': sid, 'true_label': all_labels[i],
         'predicted_label': all_preds[i], 'predicted_prob': all_probs[i]}
        for i, sid in enumerate(sample_ids)
    ]


def main():
    val_hidden_states, val_unique_ids, val_labels = [], [], []
    for path in args.val_hidden_states_path_lst:
        data = torch.load(path)
        val_hidden_states.append(data['hidden_states'])
        val_unique_ids.extend(data['unique_ids'])
        labels = data['labels'].numpy()
        bin_labels = (labels == -1).astype(int)
        val_labels.extend(bin_labels.tolist())

    val_hidden_states = torch.cat(val_hidden_states, dim=0)
    val_hidden_states_layer = val_hidden_states[:, args.layer_idx, :]

    logging.info(f"Val hidden states shape: {val_hidden_states_layer.shape}")
    arr = np.asarray(val_labels)
    logging.info(f"Val labels counts: {np.unique(arr, return_counts=True)}")

    val_dataset = TensorDataset(val_hidden_states_layer, torch.tensor(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    logging.info(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device)

    input_dim = val_hidden_states_layer.shape[1]
    model = MLPClassifier(
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, dropout=args.dropout
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.BCEWithLogitsLoss()
    final_metrics, val_probs, val_preds, val_true_labels = evaluate(
        model, val_loader, criterion, args.device
    )

    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluation Metrics  [{args.dataset_name}]")
    logging.info(f"{'='*60}")
    logging.info(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    logging.info(f"Precision: {final_metrics['precision']:.4f}")
    logging.info(f"Recall:    {final_metrics['recall']:.4f}")
    logging.info(f"F1 Score:  {final_metrics['f1']:.4f}")
    logging.info(f"AUC:       {final_metrics['auc']:.4f}")
    logging.info(f"Confusion Matrix:\n{np.array(final_metrics['confusion_matrix'])}")

    val_predictions = get_predictions(model, val_loader, args.device, val_unique_ids)

    tag = f'layer_{args.layer_idx}'
    results_path = output_dir / f'{tag}_results.json'
    with open(results_path, 'w') as f:
        json.dump({'args': vars(args), 'metrics': final_metrics,
                   'predictions': val_predictions}, f, indent=2)
    logging.info(f"Saved results to {results_path}")

    predictions_path = output_dir / f'{tag}_predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(val_predictions, f, indent=2)
    logging.info(f"Saved predictions to {predictions_path}")


if __name__ == '__main__':
    main()
