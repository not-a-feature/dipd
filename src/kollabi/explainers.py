import numpy as np
import pandas as pd
import math
import tqdm
import itertools
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from interpret.glassbox import ExplainableBoostingRegressor

from kollabi.plots import forceplot

interpret_logger = logging.getLogger('interpret')
interpret_logger.setLevel(logging.WARNING)

idx = pd.IndexSlice
# EG: vielleicht könnte man als Input Argument auch direkt den Pandas frame geben, dann brauchen wir das (...).data nicht
# GK: habe ich angepasst
# GK: nachdenken ob r2, explained variance oder adjusted r2 

class CollabExplainer:
    
    def __init__(self, df, target, test_size=0.2, verbose=False) -> None:
        self.df = df
        self.target = target
        self.fs = [col for col in df.columns if col != target]
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df[self.fs], df[target], test_size=test_size)
        self.verbose = verbose
        self.decomps = {}
        
    def new_split(self, test_size=None):
        if test_size is None:
            test_size = self.test_size
        else:
            self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.fs], self.df[self.target],
                                                                                test_size=test_size)
        self.decomps.clear()
        
    @staticmethod
    def _flip_order(comb, res):
        if comb == sorted(comb):
            return res
        else:
            res_s = res.rename({'var_f1': 'var_f2', 'var_f2': 'var_f1'}, inplace=False).copy()
            res_s = res_s.loc[res.index]
            return res_s
        
    def get(self, comb):
        comb_s = tuple(sorted(comb))
        if comb_s in self.decomps.keys():
            res = self.decomps[comb_s]
            return self._flip_order(comb, res)
        else:
            res = self.compute(list(comb_s))
            self.decomps[comb_s] = res
            return res
        
    def compute(self, comb):
        assert comb[0] in self.fs, 'Feature 1 not in the dataset'
        assert comb[1] in self.fs, 'Feature 2 not in the dataset'
        assert len(comb) == 2, 'Please provide exactly two features'
        
        return_names = ['var_f1', 'var_f2', 'additive_collab_wo_cov', '-2cov_g1_g2', 'synergetic_collab']
        
        all_numeric = all(pd.to_numeric(self.df[col], errors='coerce').notna().all() for col in comb)
        
        if all_numeric:
            y = self.df[self.target]
            var_y = np.var(y)

            X_train, X_test, y_train, y_test = train_test_split(self.df[comb], y, test_size=0.2, random_state=42)

            GAM2 = ExplainableBoostingRegressor(interactions=2)
            GAM2.fit(X_train, y_train)
            var_total = r2_score(y_test, GAM2.predict(X_test))
            #show(GAM2.explain_global())

            GAM1 = ExplainableBoostingRegressor(interactions=0)
            GAM1.fit(X_train, y_train)
            var_GAM = r2_score(y_test, GAM1.predict(X_test))
            #show(GAM1.explain_global())

            f1 = ExplainableBoostingRegressor()
            f1.fit(X_train[[comb[0]]], y_train)
            var_f1 = r2_score(y_test, f1.predict(X_test[[comb[0]]]))
            #show(f1.explain_global())

            f2 = ExplainableBoostingRegressor()
            f2.fit(X_train[[comb[1]]], y_train)
            var_f2 = r2_score(y_test, f2.predict(X_test[[comb[1]]]))
            #show(f2.explain_global())

            # getting the component funnctions g1 and g2 of the GAM
            def g1(val):
                bins = GAM1.bins_[0][0]
                if isinstance(bins, dict):
                # categorical feature
                    bin_idx = bins[val]
                else:
                # continuous feature. bins is an array of cut points
                # add 1 because the 0th bin is reserved for 'missing'
                    bin_idx = np.digitize(val, bins) + 1
                return GAM1.term_scores_[0][bin_idx]

            def g2(val):
                bins = GAM1.bins_[1][0]
                if isinstance(bins, dict):
                # categorical feature
                    bin_idx = bins[val]
                else:
                # continuous feature. bins is an array of cut points
                # add 1 because the 0th bin is reserved for 'missing'
                    bin_idx = np.digitize(val, bins) + 1
                return GAM1.term_scores_[1][bin_idx]

            cov_g1_g2 = np.cov(g1(X_test[[comb[0]]].values[:, 0]), g2(X_test[[comb[1]]].values[:, 0]))[0, 1]
            cov_g1_g2 = cov_g1_g2 / var_y
            additive_collab = (var_f1 + var_f2 - var_GAM) *-1
            additive_collab_wo_cov = additive_collab + 2*cov_g1_g2            
            synergetic_collab = var_total - var_GAM

            if self.verbose:
                print(comb)
                print('total variance Y ', var_y)
                print('test v(1 cup 2)', var_total)
                print('training v(1 cup 2): ', r2_score(y_train, GAM2.predict(X_train)))
                print('Synergetic Collaboration: ', synergetic_collab)
                print('v(', comb[0], '): ', var_f1)
                print('v(', comb[1], '): ', var_f2)

                print('Cov(g1, g2): ', cov_g1_g2 / var_y)
                print('Additive Collaboration: ', additive_collab)
                        
            return pd.Series([var_f1, var_f2, additive_collab_wo_cov, -2*cov_g1_g2, synergetic_collab], index=return_names) 
        else:
            return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=return_names)
        
    def get_all(self, only_precomputed=False, return_matrixs=False):
        '''
        Gives a detailed decomposition of all features respecting interactions and the dependencies between them

        Parameters:
            data: A datasetcontaining all the data
        '''
        logging.info('Computing all decompositions')
        if only_precomputed:
            combinations = list(self.decomps.keys())
        else:
            combinations = [list(comb) for comb in itertools.combinations(self.fs, 2)]
        
        if return_matrixs:
            vars_bivarivate = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            additive_collab = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            neg2_cov_g1_g2 = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            additive_collab_wo_cov = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            synergetic_collab = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            
            for comb in tqdm.tqdm(combinations):
                res = self.get(comb)
                # hacky but works
                vars_bivarivate.loc[comb[0], comb[0]] = res['var_f1']
                vars_bivarivate.loc[comb[1], comb[1]] = res['var_f2']  
                # rest                              
                vars_bivarivate.loc[comb[0], comb[1]] = res.sum(axis=0)
                additive_collab.loc[comb[0], comb[1]] = res['-2cov_g1_g2'] + res['additive_collab_wo_cov']
                neg2_cov_g1_g2.loc[comb[0], comb[1]] = res['-2cov_g1_g2']
                additive_collab_wo_cov.loc[comb[0], comb[1]] = res['additive_collab_wo_cov']
                synergetic_collab.loc[comb[0], comb[1]] = res['synergetic_collab']
                # make symmetric
                vars_bivarivate.loc[comb[1], comb[0]] = vars_bivarivate.loc[comb[0], comb[1]]
                additive_collab.loc[comb[1], comb[0]] = additive_collab.loc[comb[0], comb[1]]
                neg2_cov_g1_g2.loc[comb[1], comb[0]] = neg2_cov_g1_g2.loc[comb[0], comb[1]]
                additive_collab_wo_cov.loc[comb[1], comb[0]] = additive_collab_wo_cov.loc[comb[0], comb[1]]
                synergetic_collab.loc[comb[1], comb[0]] = synergetic_collab.loc[comb[0], comb[1]]
            
            return vars_bivarivate, additive_collab, synergetic_collab, neg2_cov_g1_g2, additive_collab_wo_cov
        else:                
            results = pd.DataFrame(combinations, columns=['feature1', 'feature2'])
            results.set_index(['feature1', 'feature2'], inplace=True)
            for comb in tqdm.tqdm(combinations):
                res = self.get(comb)
                results.loc[tuple(comb), res.index] = res
                res2 = self.get(comb[::-1])
                results.loc[tuple(comb[::-1]), res2.index] = res2
            return results
        
    def get_all_onefixed(self, feature):
        '''
        Gives a detailed decomposition of all features respecting interactions and the dependencies between them

        Parameters:
            data: A datasetcontaining all the data
        '''
        logging.info(f'Computing all decompositions for feature {feature}')
        combinations = [[feature, col] for col in self.fs if col != feature]
        results = pd.DataFrame(combinations, columns=['feature1', 'feature2'])
        results.set_index(['feature1', 'feature2'], inplace=True)
        for comb in tqdm.tqdm(combinations):
            res = self.get(comb)
            results.loc[tuple(comb), res.index] = res
            res_flip = res.rename({'var_f1': 'var_f2', 'var_f2': 'var_f1'})
            results.loc[tuple(comb[::-1]), res_flip.index] = res_flip
        return results
    
    def hbarplot_comb(self, comb, ax=None, figsize=None, text=True):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        with sns.axes_style('whitegrid'):
            d = self.get(comb)
            d.plot(kind='barh', ax=ax, xlabel=None, ylabel=None, use_index=False)
            plt.title(f'{comb}')
            sns.despine(left=True, bottom=True, ax=ax)
            return ax    
        
    def forceplot_onefixed(self, feature, figsize=None, ax=None, split_additive=False):
        res = self.get_all_onefixed(feature)
        ax = forceplot(res, feature, figsize=figsize, ax=ax, split_additive=split_additive)
        return ax


    def forceplots(self, figsize=(20, 10), split_additive=False, nrows=1, savepath=None):
        nplots = math.ceil(len(self.fs) / nrows)
        axss = []
        for i in range(nplots):
            # create a figure with #features subplots
            fig, axs = plt.subplots(nrows, 1, figsize=figsize)
            if nrows == 1:
                axs = [axs]
            else:
                axs = axs.flatten()
            
            fs = self.fs[i*nrows:(i+1)*nrows]
            for feature, ax in tqdm.tqdm(zip(fs, axs)):
                # handle logging
                class_level = logging.getLogger('CollabExplainer').getEffectiveLevel()
                logging.getLogger('CollabExplainer').setLevel(logging.WARNING)
                # call plot method
                self.forceplot_onefixed(feature, ax=ax, split_additive=split_additive)
                # handle logging
                logging.getLogger('CollabExplainer').setLevel(class_level)
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath + f'forceplts_{fs}.pdf')
            axss.append(axs)
        return axss
    
    def matrixplots(self, savepath=None):
        tpl = self.get_all(return_matrixs=True)
        vars_bivarivate, additive_collab, synergetic_collab, neg2_cov_g1_g2, _ = tpl
        
        cmap = sns.diverging_palette(250, 10, s=80, l=55, as_cmap=True)
        
        fig, axs = plt.subplots(2, 2, figsize=(30, 20))
        sns.heatmap(vars_bivarivate, annot=True, ax=axs[0, 0], vmin=-1, vmax=1, center=0, cmap=cmap)
        axs[0, 0].set_title('Bivariate variance')
        sns.heatmap(additive_collab, annot=True, ax=axs[1, 0], vmin=-1, vmax=1, center=0, cmap=cmap)
        axs[1, 0].set_title('Additive Collaboration')
        sns.heatmap(synergetic_collab, annot=True, ax=axs[1, 1], vmin=-1, vmax=1, center=0, cmap=cmap)
        axs[1, 1].set_title('Synergetic Collaboration')
        sns.heatmap(neg2_cov_g1_g2, annot=True, ax=axs[0, 1], vmin=-1, vmax=1, center=0, cmap=cmap)
        axs[0, 1].set_title('Negative Covariance')
        # sns.heatmap(additive_collab_wo_cov, annot=True, ax=axs[1, 2])
        # axs[1, 2].set_title('Additive Collaboration without Covariance')
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath + 'matrixplots.pdf')
        return axs
        
    def save(self, filepath):
        results = self.get_all(only_precomputed=True)
        results.to_csv(filepath)
