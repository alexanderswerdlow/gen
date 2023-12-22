from functools import partial
from typing import Any
from hydra_zen import builds, store
from hydra_zen import make_config, store
from hydra_zen.wrapper import default_to_config
from dataclasses import is_dataclass
from omegaconf import OmegaConf

def destructure(x):
    x = default_to_config(x)  # apply the default auto-config logic of `store`
    if is_dataclass(x):
        # Recursively converts:
        # dataclass -> omegaconf-dict (backed by dataclass types)
        #           -> dict -> omegaconf dict (no types)
        return OmegaConf.create(OmegaConf.to_container(OmegaConf.create(x)))  # type: ignore
    return x

def global_store(name: str, group: str, **kwargs):
    from gen.configs.base import BaseConfig
    store(
        make_config(
            hydra_defaults=["_self_"],
            bases=(BaseConfig,),
            zen_dataclass={"kw_only": True},
            **kwargs
        ),
        group=group,
        package="_global_",
        name=name,
    )
    
def stored_child_config(cls: Any, group: str, parent: str, child: str):
    store(builds(cls, override_text=False, builds_bases=(store[group][(group, parent)],)), group=group, name=child)

destructure_store = store(to_config=destructure)
auto_store = store(group=lambda cfg: cfg.name)
exp_store = partial(global_store, group="experiment")
mode_store = partial(global_store, group="experiment")