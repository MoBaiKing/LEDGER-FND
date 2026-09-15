"""Reproducible checkpoint state and explicit post-hoc scalar calibration."""
import random
import numpy as np
import torch
from torch.nn import functional as F


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def fit_scalar_temperature(logits,labels,*,calibration_ids,selection_ids,split):
    if split!='val_cal' or set(calibration_ids)&set(selection_ids):
        raise ValueError('temperature requires disjoint predefined val_cal / val_select')
    logits=logits.detach().double();labels=labels.detach().long()
    log_t=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([log_t],max_iter=50,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=F.cross_entropy(logits/log_t.exp(),labels);loss.backward();return loss
    optimizer.step(closure)
    temperature=float(log_t.detach().exp())
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('nonfinite calibration temperature')
    return dict(temperature=temperature,calibrated=True,probability_type='scalar_temperature',
                calibration_ids=sorted(calibration_ids),selection_ids=sorted(selection_ids),threshold='UNSELECTED: tune on val_select')
