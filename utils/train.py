import numpy as np


def single_train(model,
                 dataloader,
                 device,
                 logger,
                 epoch_idx,
                 global_iter_idx,
                 criterion,
                 optimizer,
                 scheduler=None,
                 metrics=[],
                 logging_freq=10,
                 writer=None):

    model.train()
    epoch_loss = []

    for idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        out = model()

        labels = data.labels
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

        if idx % logging_freq == 0:
            logger.info(
                f"Epoch: {epoch_idx:4d}, Iteration: {idx:4d} / {len(dataloader):4d} [{global_iter_idx[0]:5d}], Loss: {loss.item()}"
            )
        if writer is not None:
            writer.add_scalar('train/batch_loss', loss.item(), global_iter_idx[0])

        global_iter_idx[0] += 1

    if scheduler is not None:
        scheduler.step()

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx[0]:5d}], Epoch Loss: {epoch_loss_avg}")
    if writer is not None:
        writer.add_scalar('train/epoch_loss', epoch_loss_avg, epoch_idx)

    for metric in metrics:
        score = metric.calculate()
        metric.log(score, logger, writer=writer, global_iter=global_iter_idx[0], name_prefix='train/')

    return score, epoch_loss_avg
