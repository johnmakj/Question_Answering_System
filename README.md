[![python-version](https://img.shields.io/badge/python-v3.10-blue.svg)](https://www.python.org/)

<p float="left">
<img alt="Pandas" src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />
<img alt="NumPy" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

Question Answering System
=====

Implementation of a Questions Answering system using BM25 and fine-tuning BERT on SQuAD dataset (v1.1)

  

## Documentation
 

For a detailed installation of the project, head over to [Installation](docs/installation.md) document. 
By the end of the installation guideline, one should have a working copy of the project.

The repository contains all the necessary files to run the app. This can be done by running in terminal the command:
```bash
$ streamlit run demo/app.py
```

If you want to run each component separately, all the scripts are under the controllers directory. It contains:

* A controller for the [data preprocessing](src/controllers/data_preparation.py)
* A controller for the [model_training](src/controllers/model_training.py)
* A controller for the [model_evaluation](src/controllers/model_evaluation.py)

All these scripts can be executed from terminal. For example:
```bash
$ python src/controllers/data_preparation.py
```

Please make sure that you pass the right arguments.

In case you want to start the project from the beginning, delete data and results directory and after activating the environment
run:
```bash
$ /bin/bash build.sh
```

All the configuration parameters for data preprocessing are in [this yaml file](src/etl/config/bm25_preprocessing.yaml)
and all configuration parameters for the BERT are in [this yaml file](src/machine_learning/config/bert_config.yaml)

To run the streamlit app with your own data please change the [configuration file](demo/demo_config.yaml)