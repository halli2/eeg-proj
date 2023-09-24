import numpy as np
import numpy.typing as npt
from librosa import lpc


class LeapdModel:
    def __init__(
        self,
        y1: npt.NDArray[np.float32],
        y2: npt.NDArray[np.float32],
        order: int = 4,
    ):
        """Train classifiers for a LEAPD model.
        - y1 is a KxL matrix of preprocessed values for group 1
        - y2 ix a KxL matrix of preprocessed values for group 2"""
        self.m1, self.P1 = self._calculate(y1, order)
        self.m2, self.P2 = self._calculate(y2, order)
        self.order = order

    def _calculate(
        self,
        y: npt.NDArray[np.float32],
        order: int,
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float32]:
        """y is a KxL matrix, returns bias vector m and orthonormal basis matrix P from SVD"""
        a_all = [lpc(v, order=order) for v in y]
        b_all = [np.hstack([[0], -1 * a[1:]]) for a in a_all]
        X = np.matrix([b[1:] for b in b_all])

        m = np.transpose([np.mean(x) for x in X.T])
        D = np.diag(m)
        K = X.shape[0]
        Q = np.ones(X.shape)
        DQ = np.matmul(Q, D)
        Y = (X - DQ) / np.sqrt(K - 1)
        _U, _S, P = np.linalg.svd(Y)
        return m, P

    def classify(
        self,
        y: npt.NDArray[np.float32],
    ) -> float:
        """y is the preprocessed signal for the patient to classify
        returns the leapd index"""
        a = lpc(y, order=self.order)
        b = np.hstack([[0], -1 * a[1:]])[1:]
        D1 = self._calc_distance(b, self.m1, self.P1)
        D2 = self._calc_distance(b, self.m2, self.P2)
        rho = D2 / (D2 + D1)
        return rho

    def _calc_distance(self, a, m, P) -> float:
        a_m = np.matrix(a - m).T
        inner_sum = np.sum(
            [((p * a_m) / (p * p.T)) * p for p in P],
        )
        D = np.linalg.norm(a_m - inner_sum)
        return D
