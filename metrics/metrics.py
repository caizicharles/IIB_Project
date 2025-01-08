import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, roc_auc_score


class MetricBase():
    def __init__(self, *args, **kwargs) -> None:
        pass

    def calculate(self, probability, target):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError


class Accuracy(MetricBase):

    def __init__(self, **kwargs):
        self.NAME = 'Accuracy'

    def calculate(self, probability, target):
        probability = np.squeeze(probability, axis=-1)
        pred = (probability >= 0.5).astype(int)
        target = np.squeeze(target, axis=-1)
        return accuracy_score(target, pred)

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Sensitivity(MetricBase):

    def __init__(self, **kwargs):
        self.NAME = 'Sensitivity'

    def calculate(self, probability, target):
        probability = np.squeeze(probability, axis=-1)
        pred = (probability >= 0.5).astype(int)
        target = np.squeeze(target, axis=-1)
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Predictivity(MetricBase):

    def __init__(self, **kwargs):
        self.NAME = 'Predictivity'

    def calculate(self, probability, target):
        probability = np.squeeze(probability, axis=-1)
        pred = (probability >= 0.5).astype(int)
        target = np.squeeze(target, axis=-1)
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Negative_Predictivity(MetricBase):

    def __init__(self, **kwargs):
        self.NAME = 'Negative_Predictivity'

    def calculate(self, probability, target):
        probability = np.squeeze(probability, axis=-1)
        pred = (probability >= 0.5).astype(int)
        target = np.squeeze(target, axis=-1)
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Specificity(MetricBase):

    def __init__(self, **kwargs):
        self.NAME = 'Specificity'

    def calculate(self, probability, target):
        probability = np.squeeze(probability, axis=-1)
        pred = (probability >= 0.5).astype(int)
        target = np.squeeze(target, axis=-1)
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class AUROC(MetricBase):

    def __init__(self, **kwargs):
        self.NAME = 'AUROC'

    def calculate(self, probability, target):
        probability = np.squeeze(probability, axis=-1)
        target = np.squeeze(target, axis=-1)
        return roc_auc_score(target, probability)

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class AUPRC(MetricBase):

    def __init__(self, **kwargs):
        self.NAME = 'AUPRC'

    def calculate(self, probability, target):
        return average_precision_score(target, probability)

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


METRICS = {
    'Accuracy': Accuracy,
    'Sensitivity': Sensitivity,
    'Predictivity': Predictivity,
    'Negative_Predictivity': Negative_Predictivity,
    'Specificity': Specificity,
    'AUROC': AUROC,
    'AUPRC': AUPRC
}
