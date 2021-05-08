import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
    ## Test by running from terminal:
    ## -> python test_hydra.py optim_opts.lr=0.01 seed=3 drop_rate=0.9

    ## Multirun
    ## -> python test_hydra.py --multirun mode=segmentation,pixelcount