class EuclideanNormalization():

    def __init__(self, dim=(0, 1, 2), eps=10e-10):
        self.dim = dim
        self.eps = eps

    def __call__(self, x):
        return x / x.pow(2) \
                    .sum(dim=self.dim, keepdim=True) \
                    .clamp(min=self.eps) \
                    .sqrt()
