import pandas as pd
from kollabi.plots import forceplot

class Explanation:
    def __init__(self, title: str, scores: pd.DataFrame):
        self.title = title
        self.scores = scores
        self.plot_kwargs = {}

    def __str__(self):
        return self.title

    def __repr__(self):
        return self.title
    
    def forceplot(self, savepath=None, split_additive=False, figsize=(10, 5), **kwargs):
        # if a key is both in kwargs and self.plot_kwargs, the value in kwargs is used
        kwargs_pass = {**self.plot_kwargs, **kwargs}
        data = self.scores.transpose()
        ax = forceplot(data, self.title, split_additive=split_additive,
                       figsize=figsize, **kwargs_pass)
        if savepath is not None:
            ax.get_figure().savefig(savepath + self.title + '.pdf')
        return ax

class SurplusExplanation(Explanation):
    def __init__(self, title: str, scores: pd.DataFrame):
        super().__init__(title, scores)
        self.plot_kwargs = {'explain_surplus': True, 'rest_feature': 2}

        
class CollabExplanation(Explanation):
    def __init__(self, title: str, scores: pd.DataFrame):
        super().__init__(title, scores)
        self.plot_kwargs = {'explain_collab': True}