"""vCache decision algorithm (Schroeder et al., vCache: Verified Semantic Prompt Caching), copied verbatim from
repos/vCache/vcache/vcache_policy/strategies/verified.py (classes _Action and _Algorithm); only the imports are local."""
import random, time
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np


class sm:   # statsmodels.api.add_constant (the only statsmodels call), equivalent shim
    @staticmethod
    def add_constant(x):
        x = np.asarray(x, float)
        return np.column_stack([np.ones(len(x)), x])

from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
EmbeddingMetadataObj = object

class _Action(Enum):
    """
    Enumeration of possible actions for the algorithm.
    """

    EXPLORE = "explore"
    EXPLOIT = "exploit"


class _Algorithm:
    """
    Internal algorithm implementation for dynamic threshold computation.
    """

    def __init__(self, delta: float):
        """
        Initialize the algorithm with the given delta value.

        Args:
            delta: The delta value for confidence computation.
        """
        self.delta: float = delta
        self.P_c: float = 1.0 - self.delta
        self.epsilon_grid: np.ndarray = np.linspace(1e-6, 1 - 1e-6, 50)
        self.logistic_regression: LogisticRegression = LogisticRegression(
            penalty=None, solver="lbfgs", tol=1e-8, max_iter=1000, fit_intercept=False
        )

        self.variance_map: Dict[int, List[float]] = {
            6: 0.035445,
            7: 0.028285,
            8: 0.026436,
            9: 0.021349,
            10: 0.019371,
            11: 0.012615,
            12: 0.011433,
            13: 0.010228,
            14: 0.009963,
            15: 0.009253,
            16: 0.011674,
            17: 0.013015,
            18: 0.010897,
            19: 0.011841,
            20: 0.013081,
            21: 0.010585,
            22: 0.014255,
            23: 0.012058,
            24: 0.013002,
            25: 0.011715,
            26: 0.00839,
            27: 0.008839,
            28: 0.010628,
            29: 0.009899,
            30: 0.008033,
            31: 0.00457,
            32: 0.007335,
            33: 0.008932,
            34: 0.00729,
            35: 0.007445,
            36: 0.00761,
            37: 0.011423,
            38: 0.011233,
            39: 0.006783,
            40: 0.005233,
            41: 0.00872,
            42: 0.010005,
            43: 0.01199,
            44: 0.00977,
            45: 0.01891,
            46: 0.01513,
            47: 0.02109,
            48: 0.01531,
        }
        self.tau_latencies: List[float] = []

    def add_observation_to_metadata(
        self, similarity_score: float, is_correct: bool, metadata: EmbeddingMetadataObj
    ) -> None:
        """
        Update the metadata with the new observation

        Args
            similarity_score: float - The similarity score between the query and the embedding
            is_correct: bool - Whether the query was correct
            metadata: EmbeddingMetadataObj - The metadata of the embedding
        """
        if is_correct:
            metadata.observations.append((round(similarity_score, 3), 1))
        else:
            metadata.observations.append((round(similarity_score, 3), 0))

    def select_action(
        self, similarity_score: float, metadata: EmbeddingMetadataObj
    ) -> _Action:
        """
        Select the action to take based on the similarity score, observations, and accuracy target

        Args
            similarity_score: float - The similarity score between the query and the embedding
            metadata: EmbeddingMetadataObj - The metadata of the embedding
        Returns
            Action - Explore or Exploit
        """
        similarity_score = round(similarity_score, 3)
        similarities: np.ndarray = np.array([obs[0] for obs in metadata.observations])
        labels: np.ndarray = np.array([obs[1] for obs in metadata.observations])

        if len(similarities) < 6 or len(labels) < 6:
            return _Action.EXPLORE

        t_hat, gamma, var_t = self._estimate_parameters(
            similarities=similarities, labels=labels
        )

        if t_hat == -1:
            return _Action.EXPLORE
        metadata.gamma = gamma
        metadata.t_hat = t_hat
        metadata.var_t = var_t

        start_time = time.time()
        tau: float = self._get_tau(
            var_t=var_t, s=similarity_score, t_hat=t_hat, metadata=metadata
        )
        latency = time.time() - start_time

        # Uncomment this to save the tau latencies to a CSV file
        self.tau_latencies.append(latency)
        # if len(self.tau_latencies) % 10000 == 0:
        #    df = pd.DataFrame(self.tau_latencies, columns=['latency'])
        #    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #    print(f"Saving tau latencies to CSV: tau_latencies_{timestamp}.csv (First value: {self.tau_latencies[0]:.5f}s)")
        #    df.to_csv(f'tau_latencies_{timestamp}.csv', index=False)

        u: float = random.uniform(0, 1)
        if u <= tau:
            return _Action.EXPLORE
        else:
            return _Action.EXPLOIT

    def _estimate_parameters(
        self, similarities: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Optimize parameters with logistic regression

        Args
            similarities: np.ndarray - The similarities of the embeddings
            labels: np.ndarray - The labels of the embeddings
            metadata: EmbeddingMetadataObj - The metadata of the embedding

        Returns
            t_hat: float - The estimated threshold
            gamma: float - The estimated gamma
            var_t: float - The estimated variance of t
        """

        similarities = sm.add_constant(similarities)

        try:
            if len(similarities) != len(labels):
                print(f"len does not match: {len(similarities)} != {len(labels)}")
            self.logistic_regression.fit(similarities, labels)
            intercept, gamma = self.logistic_regression.coef_[0]

            gamma = max(gamma, 1e-6)
            t_hat = -intercept / gamma
            t_hat = float(np.clip(t_hat, 0.0, 1.0))

            similarities_col = (
                similarities[:, 1] if similarities.shape[1] > 1 else similarities[:, 0]
            )
            perfect_seperation = np.min(similarities_col[labels == 1]) > np.max(
                similarities_col[labels == 0]
            )
            var_t = self._get_var_t(
                perfect_seperation=perfect_seperation,
                n_observations=len(similarities),
                X=similarities,
                gamma=gamma,
                intercept=intercept,
            )

            return round(t_hat, 3), round(gamma, 3), var_t

        except Exception as e:
            print(f"Logistic regression failed: {e}")
            return -1.0, -1.0, -1.0

    def _get_var_t(
        self,
        perfect_seperation: bool,
        n_observations: int,
        X: np.ndarray,
        gamma: float,
        intercept: float,
    ) -> float:
        """
        Compute the variance of t using the delta method

        Args
            perfect_seperation: bool - Whether the data is perfectly separable
            n_observations: int - The number of observations
            X: np.ndarray - The design matrix
            gamma: float - The gamma parameter
            intercept: float - The intercept parameter

        Returns
            float - The variance of t

        Note:
            If the data is perfectly separable, we use the variance map to estimate the variance of t
            Otherwise, we use the delta method to estimate the variance of t
        """
        if perfect_seperation:
            if n_observations in self.variance_map:
                var_t = self.variance_map[n_observations]
            else:
                max_observations = max(self.variance_map.keys())
                var_t = self.variance_map[max_observations]
            return var_t
        else:
            p = self.logistic_regression.predict_proba(X)[:, 1]
            W = p * (1 - p)
            H = X.T @ (W[:, None] * X)

            cov_beta = np.linalg.inv(H)

            grad = np.array([-1.0 / gamma, intercept / (gamma**2)])

            var_t_hat = float(grad @ cov_beta @ grad)
            var_t_hat = max(0.0, var_t_hat)
            return var_t_hat

    def _get_tau(
        self,
        var_t: float,
        s: float,
        t_hat: float,
        metadata: EmbeddingMetadataObj,
    ) -> float:
        """
        Find the minimum tau value for the given similarity score

        Args
            var_t: float - The variance of t
            s: float - The similarity score between the query and the nearest neighbor
            t_hat: float - The estimated threshold
            metadata: EmbeddingMetadataObj - The metadata of the nearest neighbor

        Returns
            float - The minimum tau value
        """
        t_primes: List[float] = self._get_t_primes(t_hat=t_hat, var_t=var_t)
        likelihoods = self._likelihood(s=s, t=t_primes, gamma=metadata.gamma)
        alpha_lower_bounds = (1 - self.epsilon_grid) * likelihoods

        taus = 1 - (1 - self.P_c) / (1 - alpha_lower_bounds)
        metadata.t_prime = t_primes[np.argmin(taus)]
        return round(np.min(taus), 5)

    def _get_t_primes(self, t_hat: float, var_t: float) -> List[float]:
        """
        Compute all possible t_prime values.

        Args
            t_hat: float - The estimated threshold
            var_t: float - The variance of t

        Returns
            List[float] - The t_prime values
        """
        t_primes: List[float] = np.array(
            [
                self._confidence_interval(
                    t_hat=t_hat, var_t=var_t, quantile=(1 - self.epsilon_grid[i])
                )
                for i in range(len(self.epsilon_grid))
            ]
        )
        return t_primes

    def _confidence_interval(
        self, t_hat: float, var_t: float, quantile: float
    ) -> float:
        """
        Return the (upper) quantile-threshold t' such that
          P_est( t > t' ) <= 1 - quantile

        Args
            t_hat: float - The estimated threshold
            var_t: float - The variance of t
            quantile: float - The quantile

        Returns
            float - The t_prime value
        """
        z = norm.ppf(quantile)
        t_prime = t_hat + z * np.sqrt(var_t)
        return float(np.clip(t_prime, 0.0, 1.0))

    def _likelihood(self, s: float, t: float, gamma: float) -> float:
        """
        Compute the likelihood of the given similarity score and threshold

        Args
            s: float - The similarity score between the query and the nearest neighbor
            t: float - The threshold
            gamma: float - The gamma parameter

        Returns
            float - The likelihood of the given similarity score and threshold
        """
        z = gamma * (s - t)
        return expit(z)
