# behaviour_mining

## Notes on modules:
- data: recorded `.csv` files
- models: Contains self-trained model in `.zip` files introduced in the more recent versions of `stable-baselines`
- rl-baselines-zoo: Fork from `rl-baselines-zoo`for access to pretrained models
- myutils: Utils; Code from `rl-baselines-zoo.utils` with advanced code for loading pretrained models.

## Gists

1. Behaviour-mining on BipedalWalker-v2 (https://gist.github.com/philippwulff/0da6a1757e39d251d3dc6fa879538ee7)
2. Behaivor mining on pretrained models from the rl zoo:(https://colab.research.google.com/gist/ReggaeUlli/e0904f39fe47150f85d3b9172019d6ee/trained_zoo_agents.ipynb)
3. Finding distinguishing features among RL models with a RandomForestClassifier.ipynb: (https://gist.github.com/philippwulff/4d9a521254f097274b6c83a6511a9832#file-finding-distinguishing-features-among-rl-models-with-a-randomforestclassifier-ipynb)

Note: all files created during the notebook run (videos, images, tensorboard) become visible when opened in google colab.

## Dependencies

```
pip install stable-baselines
```

for box2d envs:
```
brew install swig
pip install box2d
pip install box2d-kengz
```

## Mentions

https://github.com/araffin/rl-baselines-zoo

## Links

- Difference in obs values between v2 and v3 of BipedalWalker: https://github.com/openai/gym/issues/1920

Why we should reconsider the way we generate data:
- issue mentioned here: https://stats.stackexchange.com/questions/360294/how-to-compare-two-different-algorithms-for-deep-rl
- more extensive on the issue: https://arxiv.org/abs/1709.06560

Possibly interesting for us:
- https://www.semanticscholar.org/paper/Behaviour-Mining-for-Fraud-Detection-Xu-Sung/aa79583b85fa81d50babf62eb22018b250666501
- https://www.worldscientific.com/doi/10.1142/S0219622006002271
- https://link.springer.com/article/10.1007/s00354-018-0044-4

This article states that pearson parameters should be used with doubt and KOLMOGOROV SMIRNOV CLUSTERING is superior
https://towardsdatascience.com/time-series-clustering-and-dimensionality-reduction-5b3b4e84f6a3

Trajectory comparisons of robots
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.113.461&rep=rep1&type=pdf
