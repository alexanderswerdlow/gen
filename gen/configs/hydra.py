from hydra.conf import HydraConf, JobConf, RunDir, SweepDir


def get_hydra_config():
        return HydraConf(
        run=RunDir(dir="${get_run_dir:first_level_output_path}"),
        # sweep=SweepDir(
        #     dir="${top_level_output_path}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
        #     subdir="${hydra.job.num}",
        # ),
        # job_logging=dict(
        #     version=1,
        #     formatters=dict(
        #         simple=dict(
        #             level="INFO",
        #             format="%(message)s",
        #             datefmt="[%X]",
        #         )
        #     ),
        #     handlers=dict(
        #         rich={
        #             "class": "rich.logging.RichHandler",
        #             "formatter": "simple",
        #         }
        #     ),
        #     root={"handlers": ["rich"], "level": "INFO"},
        #     disable_existing_loggers=False,
        # ),
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
    )