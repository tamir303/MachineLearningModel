import numpy as np

class ClassifierNB:
    def __init__( self ):
        self.params = {}

    def fit( self, X, y ):
        x = np.array(X)
        y = np.array(y)
        self.params[ 'cond_prob' ] = self.__prob_matrix(x, y)
        self.params[ 'cond_var' ] = np.unique(x)
        self.params[ 'class_prob' ] = self.__prob_matrix(y)
        self.params[ 'class_var' ] = np.unique(y)


    def predict( self, X ):
        x = np.array(X)
        n_samples = x.shape[ 0 ]
        predictions = [ self.__predict_sample(x[ i, : ]) for i in range(n_samples) ]

        return predictions

    def __prob_matrix( self, x, y=None ):
        M = {}

        if y is not None:
            for col_idx, cat in enumerate(x.T.tolist()):
                uniq_x = np.unique(cat)
                for x_cat in uniq_x:
                    for y_cat in np.unique(y):
                        y_samples = np.sum(y == y_cat)
                        prob = np.sum((np.array(cat) == x_cat) & (y == y_cat)) / y_samples
                        M[ (x_cat, y_cat) ] = prob
        else:
            uniq = np.unique(x)
            n_samples = len(x)
            for cat in uniq:
                M[ cat ] = np.sum(x == cat) / n_samples

        return M

    import numpy as np

    def __predict_sample( self, x ):
        class_var = self.params[ 'class_var' ]
        class_prob = self.params[ 'class_prob' ]
        cond_prob = self.params[ 'cond_prob' ]
        predictions = {}

        for cls_v in class_var:
            keys = [ (cond_v, cls_v) for cond_v in x ]
            probs = np.array([ cond_prob[ key ] for key in keys ])
            prob = np.linalg.det(np.diag(probs)) * class_prob[ cls_v ]
            predictions[ cls_v ] = prob

        # Find the class label with the maximum probability
        predicted_class = max(predictions, key=predictions.get)

        return predicted_class


class GaussianNB:
    def __init__( self ):
        self.params = {}

    def fit( self, X, y ):
        x = np.array(X)
        y = np.array(y)
        self.params['classes'] = np.unique(y)
        for c in self.params['classes']:
            x_c = x[y == c]
            self.params[c] = {
                'mean': x_c.mean(axis=0),
                'std': x_c.std(axis=0) + 1e-10,
                'prob': len(y[y == c]) / len(y)
            }

    def predict( self, X ):
        predictions = []
        for x in np.array(X):
            argmax_class_prob = self.__class_probabilities(x, argmax=True)
            predictions.append(argmax_class_prob)

        return predictions

    def __likelihod( self, x, mean, std ):
        exp = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        scalar = 1 / np.sqrt(2 * np.pi * std ** 2)

        return exp / scalar

    def __class_probabilities( self, x, argmax=False ):
        class_probabilities = {}
        for c in self.params['classes']:
            params = self.params[c]
            mean, std, cls_prob = params['mean'], params['std'], np.log(params['prob'])
            probs = self.__likelihod(x, mean, std)
            log_probs = np.sum(np.log(probs), axis=0) + cls_prob
            class_probabilities[c] = log_probs

        if argmax:
            return max(class_probabilities, key=class_probabilities.get)
        else:
            return class_probabilities
