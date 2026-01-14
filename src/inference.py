
import torch
import numpy as np

def score_anomalies(model, features, percentile):
    tensor = torch.tensor(features.values, dtype=torch.float32)
    with torch.no_grad():
        recon = model(tensor)
        errors = ((recon - tensor) ** 2).mean(dim=1).numpy()

    threshold = np.percentile(errors, percentile)
    flags = errors > threshold

    return errors.tolist(), flags.tolist(), threshold
