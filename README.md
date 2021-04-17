# behaviour_mining

## Notes on modules:
- data: recorded `.csv` files
- models: Contains self-trained model in `.zip` files introduced in the more recent versions of `stable-baselines`
- rl-baselines-zoo: Fork from `rl-baselines-zoo`for access to pretrained models
- myutils: Utils; Code from `rl-baselines-zoo.utils` with advanced code for loading pretrained models.
- notebooks: Contains Jupyter Notebooks which contain all the experiments

## Gists

1. Behaviour-mining on BipedalWalker-v3 
(https://gist.github.com/philippwulff/0da6a1757e39d251d3dc6fa879538ee7)
2. Behaivor mining on pretrained models from the rl zoo:(https://colab.research.google.com/gist/ReggaeUlli/e0904f39fe47150f85d3b9172019d6ee/trained_zoo_agents.ipynb)
3. Finding distinguishing features among RL models with a RandomForestClassifier: 
(https://gist.github.com/philippwulff/4d9a521254f097274b6c83a6511a9832#file-finding-distinguishing-features-among-rl-models-with-a-randomforestclassifier-ipynb)
4. BipedalWalker-v3-data EDA and FFT + Matrix Profile + Decomposition: 
https://gist.github.com/philippwulff/353aa5eec175101b28d06f39655abfda#file-bipedalwalker-v3-data-eda-and-fft-matrix-profile-decomposition-ipynb
5. Feature space clustering
https://colab.research.google.com/github/ReggaeUlli/BipedalWalker-gists/blob/master/Clustering_Feature_space.ipynb
6. sts-clustering with DBSCAN, OPTICS, kmeans, agglomerative on BipedalWalker-v3:
https://colab.research.google.com/gist/philippwulff/a978f96a73866b73032dd620d90e32ba/sts-clustering-with-dbscan-optics-kmeans-agglomerative-c-on-bipedalwalker-v3-obs_4.ipynb#scrollTo=migcxRI3tl2X
7. Comparison of STS against consecutive subsequence clustering + windows sizes:
https://colab.research.google.com/gist/philippwulff/93b5d65b1381caad234ded8106880ece/comparison-of-sts-against-consecutive-subsequence-clustering-windows-sizes.ipynb

8. Match clusters to videos w=30:
https://colab.research.google.com/gist/philippwulff/a1a839ba05f93f6284edc95ecb6a9611/match-clusters-to-videos-w-30.ipynb

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

This project uses code from:

https://github.com/araffin/rl-baselines-zoo
