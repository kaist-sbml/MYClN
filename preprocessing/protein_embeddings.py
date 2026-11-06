"""Actual ESM2-35M inference, matching the original cleaning and terminal cropping."""
import re
from pathlib import Path
import numpy as np

ESM_NAME = 'esm2_t12_35M_UR50D'


def clean_protein(sequence):
    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYX]', 'X', sequence.upper())
    return sequence if len(sequence) <= 1022 else sequence[:511] + sequence[-511:]


def embed_proteins(sequences, device='cpu', batch_size=16, checkpoint=None):
    import torch
    try:
        import esm
    except ImportError as exc:
        raise ImportError('Install requirements-preprocessing.txt to generate ESM embeddings') from exc
    if batch_size < 1 or not sequences or any(not s for s in sequences):
        raise ValueError('Nonempty protein sequences and batch_size >= 1 required')
    if checkpoint:
        # fair-esm's local loader expects a colocated contact-regression file.
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
    else:
        # Uses torch hub cache; downloads official ESM weights on the first run.
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    if model.embed_dim != 480 or model.num_layers != 12:
        raise ValueError('Expected esm2_t12_35M_UR50D (12 layers, 480 dimensions)')
    target = torch.device(device)
    model = model.to(target).eval()
    if target.type == 'cuda':
        model = model.half()
    convert = alphabet.get_batch_converter()
    cleaned = [clean_protein(s) for s in sequences]
    order = sorted(range(len(cleaned)), key=lambda i: len(cleaned[i]))
    result = np.empty((len(cleaned), 480), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            indices = order[start:start+batch_size]
            batch = [(str(i), cleaned[i]) for i in indices]
            _, _, tokens = convert(batch)
            tokens = tokens.to(target)
            reps = model(tokens, repr_layers=[12], return_contacts=False)['representations'][12]
            for j, index in enumerate(indices):
                result[index] = reps[j, 1:len(cleaned[index])+1].float().mean(0).cpu().numpy().astype(np.float16)
            print(f'[ESM] {start+len(indices)}/{len(order)} proteins', flush=True)
    if not np.isfinite(result).all():
        raise ValueError('ESM produced non-finite embeddings')
    return result
