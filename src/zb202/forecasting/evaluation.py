"""Frozen-model prediction and paired observed-label metrics."""
import numpy as np
import torch


def predict(model, loader, mean, scale):
    preds, truth, masks, baseline = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, current, y, mask in loader:
            preds.append(model(x, current).numpy() * scale + mean)
            truth.append(y.numpy() * scale + mean)
            masks.append(mask.numpy())
            baseline.append(np.repeat(current.numpy()[:, None, :], y.shape[1], axis=1) * scale + mean)
    return tuple(np.concatenate(v) for v in [preds, truth, masks, baseline])


def scores(pred, truth, mask):
    error = (pred - truth)[mask]
    return {'observed_labels': len(error), 'mae_ppm': float(np.abs(error).mean()) if len(error) else None,
            'rmse_ppm': float(np.sqrt((error ** 2).mean())) if len(error) else None}


