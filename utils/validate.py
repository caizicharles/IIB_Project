import torch
import numpy as np
import tqdm as tqdm


def single_validate(model, dataloader, device, logger, global_iter_idx, criterion=None, metrics=[], writer=None):

    model.eval()
    losses = []

    for _, data in enumerate(tqdm(dataloader)):
        data = data.to(device)

        with torch.no_grad():
            out = model()

            labels = data.labels
            if criterion is not None:
                loss = criterion(out, labels)
                losses.append(loss)

    loss_avg = np.mean(losses)
    logger.info(f"[{global_iter_idx[0]:5d}], Val Average Loss: {loss_avg}")

    for metric in metrics:
        score = metric.calculate()
        metric.log(score, logger, writer=writer, global_iter=global_iter_idx, name_prefix='val/')

    return score, loss_avg
