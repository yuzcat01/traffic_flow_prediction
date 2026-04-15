import numpy as np


class Evaluation:
    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        mask = target > 10
        if np.sum(mask) == 0:
            return 0.0
        return np.mean(np.abs(target[mask] - output[mask]) / target[mask])

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target, output)
        mape = Evaluation.mape_(target, output)
        rmse = Evaluation.rmse_(target, output)
        return mae, mape, rmse