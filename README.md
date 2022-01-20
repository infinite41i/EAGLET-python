# EAGLET-python

Combining multi-label classifiers based on projections of the output space using Evolutionary algorithms - based on a paper by Jose M. Moyano, Eva L. Gibaja, Krzysztof J. Cios, Sebastián Ventura

- __Original Paper:__ [https://doi.org/10.1016/j.knosys.2020.105770](https://doi.org/10.1016/j.knosys.2020.105770)
- __Original Source Code:__ [Github](https://github.com/kdis-lab/EAGLET)

## Disclaimer
__This software is written as a student project and there is no warranty for it to work and/or return correct results.__

This work is done by:
- A. Tohidi
- A. Foroutan


## How to launch EAGLET-python?

_Using a python virtual environment (venv) is suggested!_

First, make sure wheel is installed in current environment:

```bash
pip install wheel
```

Then install requirements:

```bash
pip install -r "requirements.txt"
```

__Note:__ use `--default-timeout=1000` flag if you face a `http timeout` error.

Now you can run `RunExperiment.py` with config file path as an argument. For example:

__Unix:__

```bash
python3 RunExperiment.py ./Configs/emotions_config.json
```

__Windows:__

```ps
python RunExperiment.py ./Configs/emotions_config.json
```

__Implementation References:__

1. [Datasets](http://www.uco.es/kdis/mllresources/)
1. [Paper results](https://www.uco.es/kdis/eaglet/)
1. [MLC concepts](http://scikit.ml/concepts.html)
1. [arff file extension](https://www.cs.waikato.ac.nz/ml/weka/arff.html)
1. [F-Measure](https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/)
1. [Scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
1. [Scikit-multilearn user guide](http://scikit.ml/userguide.html)
1. [scipy Sparse Matrix](https://docs.scipy.org/doc/scipy/reference/sparse.html)
