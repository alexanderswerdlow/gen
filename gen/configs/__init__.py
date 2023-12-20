from collections import namedtuple
from functools import partial
from hydra.core.config_store import ConfigStore
from gen.configs.base import BaseConfig
from gen.configs.datasets import DatasetConfig
from gen.configs.models import ModelConfig, ModelType
from gen.configs.trainer import TrainerConfig

# old
cs = ConfigStore.instance()
cs.store(name="base_config", node=BaseConfig)

# new
# from hydra_zen import (
#     MISSING,
#     ZenField,
#     builds,
#     hydrated_dataclass,
#     make_config,
# )

# def get_config_store():
#     cs = ConfigStore.instance()
#     cs.store(
#         group="hydra",
#         name="default",
#         node=dict(
#             job_logging=dict(
#                 version=1,
#                 formatters=dict(
#                     simple=dict(
#                         level="INFO",
#                         format="%(message)s",
#                         datefmt="[%X]",
#                     )
#                 ),
#                 handlers=dict(
#                     rich={
#                         "class": "rich.logging.RichHandler",
#                         "formatter": "simple",
#                     }
#                 ),
#                 root={"handlers": ["rich"], "level": "INFO"},
#                 disable_existing_loggers=False,
#             ),
#             hydra_logging=dict(
#                 version=1,
#                 formatters=dict(
#                     simple=dict(
#                         level="INFO",
#                         format="%(message)s",
#                         datefmt="[%X]",
#                     )
#                 ),
#                 handlers={
#                     "rich": {
#                         "class": "rich.logging.RichHandler",
#                         "formatter": "simple",
#                     }
#                 },
#                 root={"handlers": ["rich"], "level": "INFO"},
#                 disable_existing_loggers=False,
#             ),
#             run={
#                 "dir": "${top_level_output_path}/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}"
#             },
#             sweep={
#                 "dir": "${top_level_output_path}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
#                 "subdir": "${hydra.job.num}",
#             },
#         ),
#     )

#     zen_config = []

#     for value in BaseConfig.__dataclass_fields__.values():
#         item = (
#             ZenField(name=value.name, hint=value.type, default=value.default)
#             if value.default is not MISSING
#             else ZenField(name=value.name, hint=value.type)
#         )
#         zen_config.append(item)

#     config = make_config(
#         *zen_config,
#         hydra_defaults=[
#             "_self_",
#             dict(hydra="default"),
#             dict(dataset="huggingface"),
#             dict(model="basemapper"),
#             dict(trainer="base"),
#         ],
#     )
#     # Config
#     cs.store(name="config", node=config)
    
#     # cs.store(
#     #     group="experiment", 
#     #     package="_global_",
#     #     name="demo_exp",
#     #     node=make_config(
#     #         hydra_defaults=["_self_", {"override /model": "controlnet"}],
#     #         trainer=dict(seed=3),
#     #         profile=False,
#     #         bases=(config,),
#     #         zen_dataclass={'kw_only': True}
#     #     ),
#     # )
#     cs.store(
#         group="experiment", 
#         package="_global_",
#         name="demo_exp",
#         node=make_config(
#             hydra_defaults=["_self_"],
#             trainer=dict(seed=0, num_train_epochs=2),
#             bases=(config,),
#             zen_dataclass={'kw_only': True}
#         ),
#     )

#     cs.store(
#         group="modes", 
#         package="_global_",
#         name="test1",
#         node=make_config(
#             trainer=dict(seed=3, max_steps=10, num_epochs=5),
#             zen_dataclass={'kw_only': True},
#         ),
#     )

#     cs.store(
#         group="modes", 
#         package="_global_",
#         name="test2",
#         node=make_config(
#             trainer=dict(num_epochs=56),
#             zen_dataclass={'kw_only': True},
#             bases=(config,),
#         ),
#     )
