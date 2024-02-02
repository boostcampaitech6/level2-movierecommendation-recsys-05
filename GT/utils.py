def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}



class CFG:
    def __init__(self, config_file: str):
        extension = config_file.split(sep='.')[-1]
        if extension == 'json':
            import json
            form = json
            with open(config_file, 'r') as f:
                config = form.load(f)

        elif extension == 'yaml':
            import yaml
            form = yaml
            with open(config_file, 'r') as f:
                config = form.load(f, Loader=yaml.FullLoader)

        else:
            raise TypeError

        for key, value in config.items():
            setattr(self, key, value)
