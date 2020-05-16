# behaviour_mining

## Notes on modules:
- data: recorded `.csv` files
- models: Contains self-trained model in `.zip` files introduced in the more recent versions of `stable-baselines`
- rl-baselines-zoo: Fork from `rl-baselines-zoo`for access to pretrained models
- myutils: Utils; Code from `rl-baselines-zoo.utils` with advanced code for loading pretrained models.

## Currently working on:
- the data recording script (Ulli)
- analysing already recorded data (Philipp)

## Gists

1. Behaviour-mining on BipedalWalker-v2 (https://gist.github.com/philippwulff/0da6a1757e39d251d3dc6fa879538ee7)
2. Behaivor mining on pretrained models from the rl zoo:
https://colab.research.google.com/gist/ReggaeUlli/e0904f39fe47150f85d3b9172019d6ee/trained_zoo_agents.ipynb

Note: all files created during the notebook run (videos, images, tensorboard) become visible when opened in google colab.

## Dependencies

```
pip install stable-baselines
```

for box2d envs:
```
brew install swig
pip install box2d
pip install box2d-kengz```
