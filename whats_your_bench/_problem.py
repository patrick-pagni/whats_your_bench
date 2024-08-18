import distance

from types import SimpleNamespace
import pandas as pd

class Problem():

    def __init__(
            self,
            conjugate_prior,
            ppl_priors,
            sample_size,
            data_distribution,
            random_state
            ):
        
        self.data = data_distribution.rvs(
            size = sample_size,
            random_state = random_state
            )
        
        self.random_state = random_state

        self.conjugate_model = conjugate_prior
        self.conjugate_model.find_predictive_posterior(self.data)

        self.ppl_priors = ppl_priors

        self.models = SimpleNamespace(
            pymc_model = None,
            pyro_model = None,
            stan_model = None
        )

        self.times = SimpleNamespace(
            pymc_model = None,
            pyro_model = None,
            stan_model = None
        )

    def get_support_lim(self):
        true_params = self.conjugate_model.posterior_predictive_params

        if "mean" in true_params.__dict__.keys():
            true_params = SimpleNamespace(
                loc = true_params.mean,
                scale = true_params.cov
            )

        if "shape" in true_params.__dict__.keys():
            true_params = SimpleNamespace(
                loc = true_params.loc,
                scale = true_params.shape
            )

        if isinstance(true_params.loc, (int, float)):
            ks_lim = true_params.loc + (10*true_params.scale)
            kl_lim = [true_params.loc - (10*true_params.scale), true_params.loc + (10*true_params.scale)]

        else:
            ks_lim = (true_params.loc + (10*true_params.scale.diagonal())).max()
            kl_lim = [true_params.loc - (10*true_params.scale.diagonal()), true_params.loc + (10*true_params.scale.diagonal())]

        self.support_lim = [ks_lim, kl_lim]

    def _model_dist(self, dist, params):

        return dist(**params)

    def get_distance(self, metric, p, q, support_lim, method = 'all'):

        if metric == "ks_test":
            return distance.ks_test(p, q, support_lim, self.random_state, method = method)
        
        elif metric == "kl_divergence":
            return distance.kl_divergence(p, q, support_lim)
        
    def evaluate_models(self):

        ppl = []


        ks_distances_all = []
        ks_scores_all = []
        ks_exe_times_all = []

        ks_distances_ss = []
        ks_scores_ss = []
        ks_exe_times_ss = []

        kl_divergences = []
        kl_exe_times = []

        ppl_times = self.times.__dict__.values()

        dist = self.conjugate_model.predictive_dist

        q = self._model_dist(
            dist,
            self.conjugate_model.posterior_predictive_params.__dict__
        )

        self.get_support_lim()

        for i, model in enumerate(self.models.__dict__):

            ppl.append(model)
            p = self._model_dist(dist, getattr(self.models, model).__dict__)

            ks_results_all, ks_exe_time_all = self.get_distance(
                "ks_test",
                p,
                q,
                self.support_lim[0]
                )
            
            ks_distance_all, ks_score_all = ks_results_all
            ks_distances_all.append(ks_distance_all)
            ks_scores_all.append(ks_score_all)
            ks_exe_times_all.append(ks_exe_time_all)

            ks_results_ss, ks_exe_time_ss = self.get_distance(
                "ks_test",
                p,
                q,
                self.support_lim[0],
                method = "subsample")
            
            ks_distance_ss, ks_score_ss = ks_results_ss
            ks_distances_ss.append(ks_distance_ss)
            ks_scores_ss.append(ks_score_ss)
            ks_exe_times_ss.append(ks_exe_time_ss)

            kl_div, kl_exe_time = self.get_distance("kl_divergence", p, q, self.support_lim[1])
            kl_divergences.append(kl_div)
            kl_exe_times.append(kl_exe_time)
        
        self.results = pd.DataFrame(
            zip(
                ppl,
                ppl_times,
                ks_distances_all,
                ks_scores_all,
                ks_exe_times_all,
                ks_distances_ss,
                ks_scores_ss,
                ks_exe_times_ss,
                kl_divergences,
                kl_exe_times
                ), 
            columns = [
                "Language",
                "Model Exe Time",
                "KS Distance - All",
                "KS Score - All",
                "KS Exe Time - All",
                "KS Distance - SS",
                "KS Score - SS",
                "KS Exe Time - SS",
                "KL Divergence",
                "KL Exe Time"
                ]
            )