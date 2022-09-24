from surpyval import np
from scipy.integrate import quad


class ConvolutionModel(object):
    def __init__(self, X, Y, op='add'):
        self.X = X
        self.Y = Y

        support_X = X.dist.support
        support_Y = Y.dist.support

        # Need to double check the max support
        if op == 'add':
            self.conv = self.conv_add
            self.conv_df = self.conv_add_df
            self.l = support_X[0] + support_Y[0]
            self.h = support_X[1] + support_Y[1]
            self.ff = np.vectorize(self.ff_add)
            self.df = np.vectorize(self.df_add)
        if op == 'sub':
            self.conv = self.conv_sub
            self.conv_df = self.conv_sub_df
            self.l = support_X[0] - support_Y[1]
            self.h = support_X[1] - support_Y[0]
            self.ff = np.vectorize(self.ff_sub)
            self.df = np.vectorize(self.df_sub)

        
    def conv_add(self, x, z):
        mask = (x < self.Y.dist.support[0]) | (x > self.Y.dist.support[1])
        fy = np.where(mask, 0, self.Y.df(x))
        u = z - x
        Fx = self.X.ff(u)
        return Fx * fy

    def conv_add_df(self, x, z):
        mask = (x < self.Y.dist.support[0]) | (x > self.Y.dist.support[1])
        fy = np.where(mask, 0, self.Y.df(x))
        u = z - x
        fx = self.X.df(u)
        return fx * fy

    def conv_sub(self, x, z):
        mask = (x < self.Y.dist.support[0]) | (x > self.Y.dist.support[1])
        fy = np.where(mask, 0, self.Y.df(x))
        u = z + x
        Fx = self.X.ff(u)
        return Fx * fy

    def conv_sub_df(self, x, z):
        mask = (x < self.Y.dist.support[0]) | (x > self.Y.dist.support[1])
        fy = np.where(mask, 0, self.Y.df(x))
        u = z + x
        fx = self.X.df(u)
        return fx * fy

    def sf(self, x):
        return 1 - self.ff(x)

    def Hf(self, x):
        return -np.log(self.sf(x))

    def hf(self, x):
        return self.df(x) / self.sf(x)

    def df_add(self, x):
        return quad(lambda t : self.conv_df(t, x), self.l, x)[0]

    def ff_add(self, x):
        return quad(lambda t : self.conv(t, x), self.l, x)[0]

    def ff_sub(self, x):
        if x > 0:
            ff = quad(lambda t : self.conv(t, x), -x, 0.)[0] + quad(lambda t : self.conv(t, x), 0, np.inf)[0]
        else:
            ff = quad(lambda t : self.conv(t, x), -x, np.inf)[0]

        return ff

    def df_sub(self, x):
        if x > 0:
            ff = quad(lambda t : self.conv_df(t, x), -x, 0.)[0] + quad(lambda t : self.conv_df(t, x), 0, np.inf)[0]
        else:
            ff = quad(lambda t : self.conv_df(t, x), -x, np.inf)[0]

        return ff