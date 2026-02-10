from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        if isinstance(x_t, tuple):
            return tuple((1 + cfg_strength) * p - cfg_strength * np for p, np in zip(pred, neg_pred))
        else:
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
