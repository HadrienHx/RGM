import numpy as np
import scipy.linalg


class LinearRegression(object):
    def __init__(self, dataset, reg):
        self.dataset = dataset
        self.X = dataset.data # n x d
        self.y = dataset.targets
        self.reg = reg

        self.n = len(self.X)
        self.d = self.X.shape[1]

        self.A = np.array(self.X.T @ self.X) / self.n + self.reg * np.diag(np.ones(dataset.d))
        self.b = np.array(self.X.T @ self.y).flatten() / self.n

        print(f"Eigenvalues of A above reg: {np.mean((np.linalg.eigvalsh(np.array(self.X.T @ self.X) / self.n) > reg))}")
            
        self.bis = [x * y for x,y in zip(self.X, self.y)] 
        self.Ainv = None


    def get_max_radius(self):
        dmax = 0
        Ainv = self.get_Ainv()

        for X_i in self.X:
            X_i = np.array(X_i).flatten()
            dmax = max(dmax, X_i @ Ainv @ X_i) 

    def get_max_radius2(self):
        dmax = 0
        Ainv = self.get_Ainv()

        for X_i in self.X:
            X_i = np.array(X_i).flatten()    
            d2 = np.sqrt(X_i @ X_i * X_i @ Ainv @ Ainv @ X_i) 
            dmax = max(dmax, d2)

    def get_Ainv(self):
        if self.Ainv is None:
            self.Ainv = np.linalg.pinv(self.A)
        return self.Ainv

    def get_solution(self):
        return self.get_Ainv() @ self.b
    
    def get_gradient(self, x):
        return self.A @ x - self.b

    def get_individual_gradients(self, x):
        return np.array([(Xi * (Xi @ x - yi) + self.reg * x) / self.n for Xi, yi in zip(np.array(self.X), self.y)])

    def get_smoothness(self):
        return np.max(np.linalg.eigvalsh(self.A))

    def get_largest_Ai(self):
        return max([np.linalg.norm(x)**2 / self.n for x in self.X])
    
    def get_error(self, x):
        return 0.5 * np.linalg.norm(self.X @ x - self.y)**2 / self.n + 0.5 * self.reg * x @ x

    def copy(self, dataset):
        return LinearRegression(dataset, self.reg)

class TiltedLinearRegression(LinearRegression):
    def __init__(self, dataset, reg, tilt):
        self.tilt = tilt
        super().__init__(dataset, reg)

    def get_solution(self):
        return self.get_Ainv() @ self.b + self.tilt
    
    def get_gradient(self, x):
        return super().get_gradient(x - self.tilt)

    def get_individual_gradients(self, x):
        return super().get_individual_gradients(x - self.tilt)
    
    def get_error(self, x):
        return super().get_error(x - self.tilt)
    
    def copy(self, dataset):
        return TiltedLinearRegression(dataset, self.reg, self.tilt)