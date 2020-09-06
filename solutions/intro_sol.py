# Solutions for Intro tutorial - PyData Global 2020 Tutorial

class Diagonal(LinearOperator):
    """Short version of a Diagonal operator. See
    https://github.com/equinor/pylops/blob/master/pylops/basicoperators/Diagonal.py
    for a more detailed implementation
    """
    def __init__(self, diag, dtype='float64'):
        self.diag = diag.flatten()
        self.shape = (len(self.diag), len(self.diag))
        self.dtype = np.dtype(dtype)

    def _matvec(self, x):
        y = self.diag*x
        return y

    def _rmatvec(self, x):
        y = self.diag*x
        return y


class FirstDerivative(LinearOperator):
    """Short version of a FirstDerivative operator. See
    https://github.com/equinor/pylops/blob/master/pylops/basicoperators/FirstDerivative.py
    for a more detailed implementation
    """
    def __init__(self, N, sampling=1., dtype='float64'):
        self.N = N
        self.sampling = sampling
        self.shape = (N, N)
        self.dtype = dtype
        self.explicit = False

    def _matvec(self, x):
        x, y = x.squeeze(), np.zeros(self.N, self.dtype)
        y[1:-1] = (0.5 * x[2:] - 0.5 * x[0:-2]) / self.sampling
        # edges
        y[0] = (x[1] - x[0]) / self.sampling
        y[-1] = (x[-1] - x[-2]) / self.sampling
        return y

    def _rmatvec(self, x):
        x, y = x.squeeze(), np.zeros(self.N, self.dtype)
        y[0:-2] -= (0.5 * x[1:-1]) / self.sampling
        y[2:] += (0.5 * x[1:-1]) / self.sampling
        # edges
        y[0] -= x[0] / self.sampling
        y[1] += x[0] / self.sampling
        y[-2] -= x[-1] / self.sampling
        y[-1] += x[-1] / self.sampling
        return y


def Diagonal_timing():
    """Timing of Diagonal operator
    """
    n = 10000
    diag = np.arange(n)
    x = np.ones(n)

    # dense
    D = np.diag(diag)

    from scipy import sparse
    Ds = sparse.diags(diag, 0)

    # lop
    Dop = Diagonal(diag)

    # uncomment these
    #%timeit -n3 -r3 np.dot(D, x)
    #%timeit -n3 -r3 Ds.dot(x)
    #%timeit -n3 -r3 Dop._matvec(x)


def FirstDerivative_timing():
    """Timing of FirstDerivative operator
    """
    nx = 2001
    x = np.arange(nx) - (nx-1)/2

    # dense
    D = np.diag(0.5*np.ones(nx-1),k=1) - np.diag(0.5*np.ones(nx-1),-1)
    D[0, 0] = D[-1, -2] = -1
    D[0, 1] = D[-1, -1] = 1

    # lop
    Dop = pylops.FirstDerivative(nx, edge=True)

    # uncomment these
    # %timeit -n3 -r3 np.dot(D, x)
    # %timeit -n3 -r3 Dop._matvec(x)


def FirstDerivative_memory():
    """Memory footprint of Diagonal operator
    """
    from pympler import asizeof
    from scipy.sparse import diags
    nn = (10 ** np.arange(2, 4, 0.5)).astype(np.int)

    mem_D = []
    mem_Ds = []
    mem_Dop = []
    for n in nn:
        D = np.diag(0.5 * np.ones(n - 1), k=1) - np.diag(0.5 * np.ones(n - 1),
                                                         -1)
        D[0, 0] = D[-1, -2] = -1
        D[0, 1] = D[-1, -1] = 1
        Ds = diags((0.5 * np.ones(n - 1), -0.5 * np.ones(n - 1)),
                   offsets=(1, -1))
        Dop = pylops.FirstDerivative(n, edge=True)
        mem_D.append(asizeof.asizeof(D))
        mem_Ds.append(asizeof.asizeof(Ds))
        mem_Dop.append(asizeof.asizeof(Dop))

    plt.figure(figsize=(12, 3))
    plt.semilogy(nn, mem_D, '.-k', label='D')
    plt.semilogy(nn, mem_Ds, '.-b', label='Ds')
    plt.semilogy(nn, mem_Dop, '.-r', label='Dop')
    plt.legend()
    plt.title('Memory comparison')