from distance import kl_divergence

class Problem():

    def __init__(self, priors, posterior, data):
        """
        Init function for Problem class

        Args:
            priors: Prior distribution for PPL
            posterior: True value for posterior given prior and data
            data: Data from which posterior has been estimated.
        """
        self.priors = priors
        self.posterior = posterior
        self.data = data



    def distance(self, predicted, true, metric = None):
        # TODO: Establish procedure for defining dynamic support range depending on PDF
        # TODO: Ensure predicted and true are PDFs
        assert metric is not None

        if metric = "kl_div":
            return kl_divergence(true, predicted, -10, 10)
