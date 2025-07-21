import torch.nn as nn

from .nn.modules import (
    InducingLinear,
    InducingConv1d,
    InducingConv2d,
    InducingConv3d,
)

# Original layer type → Bayes layer type mapping
REPLACEMENTS = {
    nn.Linear:    InducingLinear,
    nn.Conv1d:    InducingConv1d,
    nn.Conv2d:    InducingConv2d,
    nn.Conv3d:    InducingConv3d,
}

def bayesianize_(model: nn.Module, config: dict, prefix: str = ""):
    """
    Recursively replace Linear/Conv* layers in model with Inducing versions.
    config format reference:
    {
      "inference": {
        "Conv2d": {"inference":"inducing","inducing_rows":128,"inducing_cols":128},
        "Linear": {"inference":"inducing","inducing_rows":100,"inducing_cols":128}
      },
      "whitened_u": True,
      "q_inducing": "diagonal",
      "learn_lamda": True,
      "init_lamda": 0.001,
      "max_lamda": 0.03,
      "max_sd_u": 0.1,
      "cache_cholesky": True,
      "prior_sd": 1.0,
      "sqrt_width_scaling": True,
      "key_layers": ["layer2.1.conv2", "layer4.1.conv2", "fc"]
    }
    """
 
    per_layer_cfg = config.get("inference", {})
  
    global_keys = [
        "whitened_u", "q_inducing", "learn_lamda", "init_lamda",
        "max_lamda", "max_sd_u", "cache_cholesky",
        "prior_sd", "sqrt_width_scaling", "key_layers"
    ]
    global_cfg = {k: config[k] for k in global_keys if k in config}
   
    for name, child in list(model.named_children()):
        child_cls = type(child)
     
        full_name = f"{prefix}.{name}" if prefix else name
        # print(full_name)
        bayesianize_(child, config, prefix=full_name)
     
  
        if child_cls in REPLACEMENTS and full_name in config["key_layers"]:
           
            bayes_cls = REPLACEMENTS[child_cls]
            layer_name = child_cls.__name__ 

           
            local = per_layer_cfg.get(layer_name, {})
            layer_local = {
                k: local[k]
                for k in ("inducing_rows", "inducing_cols")
                if k in local
            }

         
            layer_kwargs = {**global_cfg, **layer_local}

           
            if isinstance(child, nn.Linear):
          
                ctor_args   = [child.in_features, child.out_features]
                ctor_kwargs = {"bias": (child.bias is not None)}
                
            else:
              
                ctor_args = [
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                ]
                ctor_kwargs = {
                    "stride":        child.stride,
                    "padding":       child.padding,
                    "dilation":      child.dilation,
                    "groups":        child.groups,
                    "bias":          (child.bias is not None),
                    "padding_mode":  child.padding_mode,
                }

            # 4) Add mixin parameters and instantiate
            ctor_kwargs.update(layer_kwargs)
            new_layer = bayes_cls(*ctor_args, **ctor_kwargs)

            # 5) Replace back to model
            setattr(model, name, new_layer)

            # (Optional) print check
            # print(f"[Bayesize] {name}: "
            #       f"{layer_name} → {bayes_cls.__name__}  "
            #       f"inducing=({new_layer.inducing_rows}, {new_layer.inducing_cols})")
