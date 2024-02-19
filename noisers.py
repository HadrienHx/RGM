import numpy as np 
import utils
from dataset import Dataset


class PrivateAlgorithm(object):
    def __init__(self, model, alpha, epsilon):
        self.model = model 
        self.alpha = alpha
        self.epsilon = epsilon

        self.initialize_params()

    def get_new_param(self, x, lr):
        raise NotImplementedError
    
    def initialize_params(self):
        pass


class NoNoiser(PrivateAlgorithm):
    name = "NoNoise"

    def get_new_param(self, x, lr):
        return x - lr * self.model.get_gradient(x)


class UnboundedNoiser(PrivateAlgorithm):
    name = "RelNoiser"

    def __init__(self, model, alpha, epsilon, quantile=1., filter=1, remove_mean=True):
        self.delta = None
        self.R_star = None 
        self.quantile = quantile
        self.filter = filter
        self.remove_mean = remove_mean
        self.flag = True
        super().__init__(model, alpha, epsilon)

    def get_new_param(self, x, lr):
        if not self.flag:
            return x

        g = self.model.get_gradient(x)
        return x - lr * np.random.normal(g, self.gamma * np.linalg.norm(g) + self.sigma)

    def initialize_params(self):
        self.filter_outliers(self.filter)
        self.get_delta()
        self.get_R_star()

        if self.delta > 1: 
            print(f"Delta > 1")
            self.flag = False

        print(f"delta: {self.delta}, R_star: {self.R_star}")

        self.gamma =  self.delta * np.sqrt(self.alpha / self.epsilon)
        self.sigma = self.gamma * self.R_star / self.delta

    def print_pessimistic_delta(self):
        pessimistic_delta = max([ 
            (x @ x.T)[0,0] / self.model.n
            for x in self.model.X
        ]) / min(np.linalg.eigvalsh(self.model.A))

        print(f"pessimistic delta: {pessimistic_delta}")

    def filter_outliers(self, k):
        if k == 0:
            return 
        x_bar = 0.
        if self.remove_mean:
            x_bar = np.average(self.model.X, axis=0)
        A_inv_sq = self.model.get_Ainv() @ self.model.get_Ainv()
        deltas = np.array([ 
            (((x - x_bar) @ (x - x_bar).T) @ ((x- x_bar) @ A_inv_sq @ (x - x_bar).T))[0,0] / self.model.n
            for x in self.model.X
        ])

        delta_q = np.quantile(deltas, self.quantile)

        clip_factors = np.array([np.power(delta_q / max(d, delta_q), 1./4) for d in deltas])
        
        new_X = np.matrix(np.array([(x_bar + (x - x_bar) * cf)[0] for x, cf in zip(self.model.X, clip_factors)]).reshape((self.model.n, self.model.d)))

        self.model = self.model.copy(Dataset(data=new_X, targets=self.model.dataset.targets * clip_factors))
        self.filter_outliers(k-1)

    def get_delta(self):
        x_bar = 0.
        if self.remove_mean:
            x_bar = np.average(self.model.X, axis=0)
        sqAinv = self.model.get_Ainv() @ self.model.get_Ainv()
        self.delta = max([ 
            (np.sqrt(((x - x_bar) @ (x - x_bar).T) @ ((x - x_bar) @ sqAinv @ (x - x_bar).T)) / self.model.n)[0,0]
            for x in self.model.X
        ])

    def get_optimistic_R_star(self):
        n_samples = 1000
        pairs = np.random.randint(0, len(self.model.X), size=(n_samples, 2))
        indiv_grads = self.model.get_individual_gradients(self.model.get_solution())
        self.R_star = max(
            [np.linalg.norm(indiv_grads[i] - indiv_grads[j]) for i,j in pairs])


    def get_R_star(self):
        max_norm_bis = max([np.linalg.norm(x) for x in self.model.bis])
        # print(f"Norms ratio b/max_bi: {np.linalg.norm(self.model.b) / max_norm_bis}")
        self.R_star = self.delta * np.linalg.norm(self.model.b) + max_norm_bis / self.model.n 


class ClippingNoiser(PrivateAlgorithm):
    name = "ClipNoiser"
    def __init__(self, model, alpha, epsilon, c=None, c_fac=1.):
        self.c = c
        self.c_fac = c_fac
        if c is None: 
            print(f"No parameter c passed to ClippingNoiser")

        self.sigma = None
        super().__init__(model, alpha, epsilon)

    def initialize_params(self):
        if self.c is None:
            self.autoset_clipping_threshold()
            print(f"Defaulting clipping threshold to: {self.c}")
        self.sigma = self.c * np.sqrt(self.alpha / self.epsilon)

    def get_new_param(self, x, lr):
        g = np.sum([
            utils.clip(gi, self.c) for gi in
            self.model.get_individual_gradients(x)
        ], axis=0)
        return x - lr * np.random.normal(g, self.sigma)
    
    def autoset_clipping_threshold(self):
        x_star = self.model.get_solution()
        self.c = self.c_fac * max([np.linalg.norm(self.model.get_individual_gradients(x_star))])

