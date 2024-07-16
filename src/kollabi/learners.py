import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, lineno=738, message='Missing values detected.')

from interpret.glassbox import ExplainableBoostingRegressor

class Predictor:
    def __init__(self, interactions=0.95, exclude=None):
        self.interactions = interactions
        self.exclude = exclude
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

    def predict(self, X, **kwargs):
        fs = sorted(list(X.columns))
        return self.model.predict(X.loc[:, fs], **kwargs)

    def fit(self, X, y):
        fs = sorted(list(X.columns))
        self.model.fit(X.loc[:, fs], y)

    def predict_component(self, X, component):
        pass

    def predict_components(self, X, components):
        return sum([self.predict_component(X, c) for c in components])


from sklearn.linear_model import LinearRegression, Ridge


class BivariateLinearPredictor(Predictor):
    def __init__(self, type='ols', interactions=0):
        super().__init__(interactions=interactions)
        assert interactions == 0, 'Higher order interactions not supported for this model class'
        self.type = type
        models = {
            'ols': LinearRegression,
            'ridge': Ridge
        }
        if self.type in models.keys():
            self.model = models[self.type]()
        else:
            raise NotImplementedError('Only {} supported for now'.format(', '.join(models.keys())))

    def predict_component(self, X, component):
        if isinstance(component, str):
            assert component in X.columns
            variable_index = X.columns.tolist().index(component)
            coef = self.model.coef_[variable_index]
            return coef * X.loc[:, component]
        else:
            raise NotImplementedError('No higher order interactions supported for this model class')


from interpret.utils._clean_x import preclean_X
from interpret.glassbox._ebm._bin import ebm_eval_terms
                
class EBM(Predictor):
    
    def __init__(self, interactions=0.95, exclude=None):
        super().__init__(interactions=interactions, exclude=exclude)
        self.model = ExplainableBoostingRegressor(interactions=interactions, exclude=exclude)
                        
    def predict_components(self, X, components):
        """
        Due to limitations of the interpret package we can query multiple components at once,
        but can only get the aggregation of the components at once, not the individual contributions.
        To get the individual contributions we need to query each component separately.
        """
        comp_names = []
        comp_ixs = []
        for component in components:
            if isinstance(component, str):
                comp_name = component
            elif isinstance(component, list) or isinstance(component, tuple):
                # component = list(component)
                component = sorted(component, key=X.columns.tolist().index)
                comp_name = ' & '.join(component)
            else:
                raise NotImplementedError('only str or list of strings supported for component')
            try:  
                comp_index = self.model.term_names_.index(comp_name)
                comp_names.append(comp_name)
                comp_ixs.append(comp_index)
            except Exception as err:
                print(err)
                print(f'Probably, component {comp_name} was not found in the model')
            
        comp_ixs = np.array(comp_ixs).astype(int)
        
        # taken from the interpret package
        X, n_samples = preclean_X(X, self.model.feature_names_in_, self.model.feature_types_in_)
        n_scores = 1 if isinstance(self.model.intercept_, float) else len(self.model.intercept_)
        explanations = ebm_eval_terms(
            X,
            n_samples,
            n_scores,
            self.model.feature_names_in_,
            self.model.feature_types_in_,
            self.model.bins_,
            [self.model.term_scores_[comp_ix] for comp_ix in comp_ixs],
            [self.model.term_features_[comp_ix] for comp_ix in comp_ixs],
        )        
        
        return np.mean(explanations, axis=1)
    
    def predict_component(self, X, component):
        return self.predict_components(X, [component])

