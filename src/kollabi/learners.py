import pickle


class Predictor:
    def __init__(self):
        self.model = None

    @staticmethod
    def load(s):
        predictor = Predictor()
        predictor.model = pickle.loads(s)
        return predictor

    def pickle(self):
        s = pickle.dumps(self.model)
        return s

    def save(self, path):
        pass

    def predict(self, X):
        pass

    def fit(self, X, y):
        pass

    def predict_component(self, X, component):
        pass

    def predict_components(self, X, components):
        return sum([self.predict_component(X, c) for c in components])


from sklearn.linear_model import LinearRegression, Ridge


class BivariateLinearPredictor(Predictor):
    def __init__(self, type='ols'):
        super().__init__()
        self.type = type
        models = {
            'ols': LinearRegression,
            'ridge': Ridge
        }
        if self.type in models.keys():
            self.model = models[self.type]()
        else:
            raise NotImplementedError('Only {} supported for now'.format(', '.join(models.keys())))

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_component(self, X, component):
        if isinstance(component, str):
            assert component in X.columns
            variable_index = X.columns.tolist().index(component)
            coef = self.model.coef_[variable_index]
            return coef * X.loc[:, component]
        else:
            raise NotImplementedError('No higher order interactions supported for this model class')