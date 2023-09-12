import numpy as np


class PCA:
    def __init__( self, num_features ):
        self.params = {
            'nf': num_features
        }

    def fit_transform( self, X ):
        x = np.array(X)
        self.params[ 'm' ] = np.mean(x, axis=0)
        self.params[ 'sd' ] = np.std(x, axis=0)
        standard_x = self.__standardize(x)
        self.params[ 'cov' ] = cov = self.__cov_mat(standard_x)
        eigenvalues, eigenvectors = self.__desc_sorted_eigen(cov)
        self.params[ 'pca' ] = z_pca, _ = self.__project(eigenvectors, eigenvalues, standard_x)

        return z_pca

    def __standardize( self, x ):
        mean = self.params[ 'm' ]
        sd_dev = self.params[ 'sd' ]
        standard_x = (x - mean) / sd_dev

        return standard_x

    def __cov_mat( self, x ):
        cov_func = lambda x1, x2, m1, m2: np.dot(np.transpose(x1 - m1), (x2 - m2)) / max(n_samples - 1, 1)
        n_samples, n_features = x.shape
        means = self.params[ 'm' ]
        cov = np.array(
            [ [ cov_func(x[ :, x1 ], x[ :, x2 ], means[ x1 ], means[ x2 ]) for x1 in range(n_features) ]
              for x2 in range(n_features) ],
            dtype=float
        )

        return cov

    def __desc_sorted_eigen( self, cov ):
        # Sort EigenValues indexes by higher values
        # Sort EigenVectors according to EigenValues indexes
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = eigenvalues.argsort()[ ::-1 ]
        eigenvalues = eigenvalues[ idx ]
        eigenvectors = eigenvectors[ :, idx ]

        return eigenvalues, eigenvectors

    def __project( self, eig_vec, eig_val, standard_x ):
        num_features = self.params[ 'nf' ]
        u = eig_vec[ :, :num_features ]
        z_pca = np.dot(standard_x, u)

        total_variation = np.sum(eig_val)
        var_pca = eig_val / total_variation * 100

        return z_pca, var_pca
