from neuroflex.utils.experiment_pipeline.config import (Config)

if __name__ == "__main__":
    config = Config("/neuroflex/utils/CONFIG_LOCAL.json")

    for key in config.__dict__.keys():
        print(key, config.__dict__[key])

    print(config.get("original_model_id"))
    config.start_experiment()

    print(config.get("begin_time"))