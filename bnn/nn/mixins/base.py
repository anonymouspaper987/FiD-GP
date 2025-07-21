from abc import ABC, abstractmethod

import torch.nn as nn


class BayesianMixin(ABC, nn.Module):

    @abstractmethod
    def parameter_loss(self):
        """Calculates generic parameter-dependent loss. For a probabilistic module with some prior over the parameters,
        e.g. for MAP inference or MCMC sampling, this would be the negative log prior, for Variational inference the
        KL divergence between approximate posterior and prior."""
        raise NotImplementedError

  
