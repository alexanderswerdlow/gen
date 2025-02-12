from __future__ import annotations

import inspect
from dataclasses import is_dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Union, get_type_hints

from hydra import compose, initialize
from hydra_zen import builds, make_config, store
from hydra_zen.wrapper import default_to_config
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig

def destructure(x):
    x = default_to_config(x)  # apply the default auto-config logic of `store`
    if is_dataclass(x):
        # Recursively converts:
        # dataclass -> omegaconf-dict (backed by dataclass types)
        #           -> dict -> omegaconf dict (no types)
        return OmegaConf.create(OmegaConf.to_container(OmegaConf.create(x)))  # type: ignore
    return x


destructure_store = store(to_config=destructure)


def global_store(name: str, group: str, hydra_defaults: Optional[list[Any]] = None, **kwargs):
    from gen.configs.base import BaseConfig

    cfg = make_config(
        hydra_defaults=hydra_defaults if hydra_defaults is not None else ["_self_"],
        bases=(BaseConfig,),
        zen_dataclass={"kw_only": True},
        **kwargs,
    )
    destructure_store(
        cfg,
        group=group,
        package="_global_",
        name=name,
    )
    return cfg

def store_child_config(cls: Any, group: str, parent: str, child: str, **kwargs):
    store(builds(cls, builds_bases=(store[group][(group, parent)],)), group=group, name=child, **kwargs)


auto_store = store(group=lambda cfg: cfg.name)
exp_store = partial(global_store, group="experiment")
mode_store = partial(global_store, group="modes")

def inherit_parent_args(cls):
    parent = cls.__bases__[0]  # Assuming first parent is the one we want to inherit args from
    parent_params = inspect.signature(parent.__init__).parameters
    child_init = cls.__init__

    def new_init(self, **kwargs):
        # Extract parent and child args based on parameter names
        parent_kwargs = {k: v for k, v in kwargs.items() if k in parent_params}
        child_kwargs = {k: v for k, v in kwargs.items() if k not in parent_params}

        super(cls, self).__init__(**parent_kwargs)
        child_init(self, **child_kwargs)

    child_params = list(inspect.signature(child_init).parameters.values())
    new_init.__signature__ = inspect.Signature(
        parameters=[child_params[0]] + list(parent_params.values())[1:] + child_params[1:], return_annotation=inspect.Signature.empty
    )
    new_init.__annotations__ = {**get_type_hints(parent.__init__), **get_type_hints(child_init)}

    cls.__init__ = new_init
    return cls

def get_cfg(overrides: Union[str, list[str]], handle_dtype: bool = True) -> BaseConfig:
    with initialize(config_path=None, version_base=None):
        cfg = compose(config_name="config", return_hydra_config=True, overrides=overrides)

        return cfg