# Hydra example
This repository acts as an example ML project that utilizes hydra configuration.
Hydra provides yaml-based, hierarchical configuration with compositions and overrides built-in.
`configs/` directory contains grouped configs for the code used in `src/` directory. 

Key features:

- Hierarchical configuration composable from multiple sources
- Configuration can be specified or overridden from the command line
- Dynamic command line tab completion
- Run your application locally or launch it to run remotely
- Run multiple jobs with different arguments with a single command

### Run example training
```bash
pytorch-hydra-example$ . ./activate_env.sh
(env) pytorch-hydra-example$ python src/train.py 
```

### Run example multirun training
You can run your code multiple times with the `--multirun` flag 
```bash
pytorch-hydra-example$ . ./activate_env.sh
(env) pytorch-hydra-example$ python src/train.py --multirun dataset=cifar10, mnist # runs training on two datasets sequentially
```


### Override default parameters
You can easily override parameters specified in hydra config `yaml` files
```bash
pytorch-hydra-example$ . ./activate_env.sh
(env) pytorch-hydra-example$ python src/train.py batch_size=256
```

### Project structure
```bash
.
├── activate_env.sh                     # Script for activating project environment
├── configs                             # Hydra configs
│   ├── dataset                             # Dataset configs
│   │   ├── cifar10.yaml
│   │   └── mnist.yaml
│   ├── model                               # ML model configs
│   │   └── lenet5.yaml
│   ├── train.yaml                          # Training config
│   └── transform                           # Data transform configs
│       └── default_transforms.yaml
└── src                                 # Source code
    ├── __init__.py
    ├── models                              # ML model architectures
        ├── __init__.py
    │   └── lenet5.py
    └── train.py                            # Train script
├── requirements.txt
└── setup.py
```