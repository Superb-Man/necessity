# working from place
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def test_model(model, test_generator, loss_fn, device='cuda'):

    model.eval()
    raw_predictions = defaultdict(list)
    loss_overall = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_generator):
            signal, _, rhythm, patient_ids = batch 
            signal, rhythm = signal.to(device, non_blocking=True), rhythm.to(device, non_blocking=True)

            output_rhy = model(signal)
            loss_batch = loss_fn(output_rhy, rhythm.argmax(dim=1))

            loss_overall += loss_batch.item()
            num_batches += 1

            predicted_labels = torch.argmax(output_rhy, dim=1).cpu().numpy()
            patient_ids = patient_ids.cpu().numpy().flatten()

            for pid, pred in zip(patient_ids, predicted_labels):
                raw_predictions[pid].append(pred) # mappinng each result to its corresponding patient id

    return raw_predictions, loss_overall / num_batches


def apply_majority_voting(predictions):

    final_predictions = {}
    for pid, preds in predictions.items():
        final_predictions[pid] = max(set(preds), key=preds.count) 

    return final_predictions
