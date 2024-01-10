from hydra.conf import RunDir, SweepDir


def get_hydra_config():
    return dict(
        job_logging=dict(
            version=1,
            formatters=dict(
                simple=dict(
                    level="INFO",
                    format="%(message)s",
                    datefmt="[%X]",
                )
            ),
            handlers=dict(
                rich={
                    "class": "rich.logging.RichHandler",
                    "formatter": "simple",
                }
            ),
            root={"handlers": ["rich"], "level": "INFO"},
            disable_existing_loggers=False,
        ),
        hydra_logging=dict(
            version=1,
            formatters=dict(
                simple=dict(
                    level="INFO",
                    format="%(message)s",
                    datefmt="[%X]",
                )
            ),
            handlers={
                "rich": {
                    "class": "rich.logging.RichHandler",
                    "formatter": "simple",
                }
            },
            root={"handlers": ["rich"], "level": "INFO"},
            disable_existing_loggers=False,
        ),
        run={"dir": "${top_level_output_path}/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}"},
        sweep={
            "dir": "${top_level_output_path}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
            "subdir": "${hydra.job.num}",
        },
    )

    #     run=RunDir(dir="${top_level_output_path}/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}"),
    #     sweep=SweepDir(dir="${top_level_output_path}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}", subdir="${hydra.job.num}"),
