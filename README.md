# Type 2 Diabetes Mellitus Neural Network (DiabNet)

A Neural Network to predict type 2 diabetes (T2D) using a collection of SNPs highly correlated with T2D.

## Installation & Configuration

To install the latest version, clone this repository:

```bash
git clone https://github.com/zgcarvalho/diabnet.git
```

Install [poetry](https://python-poetry.org/) with [pip](https://pip.pypa.io/en/stable/):

```bash
pip install poetry
```

Configure poetry to create a virtual environment inside the project:

```bash
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
```

Install dependencies:

```bash
poetry install
```

To start the virtual environment, run:

```bash
poetry shell
```

## DiabNet SNPs datasets

There are three sets with 1000 SNPs, each has been defined:

- associated SNPs;
- not associated SNPs;
- random SNPs or randomly choosen SNPs.

Besides these sets, we made another four derived from the associated SNPs set, that are:

- shuffled labels;
- shuffled ages;
- shuffled parent diagnosis;
- shuffled SNPs or shuffled associated SNPs;
- family exclusion.

These shuffled sets were used to analyze the importance of some features, e. g. labels, ages, parent diagnosis and SNPs. The great capacity of neural networks to fit data during training are well known. Thus, we are looking for the impact of this artificial noise on the inference of the validation subset.

The shuffled associated SNPs set was created to preserve the observed frequency for each SNP. With that, we reduced a possible bias present in the regularization parameters that could affect the training using both non-associated and random SNPs sets.

## Training DiabNet

The DiabNet training is done via training.py, using a configuration file.

### Simple training

To train DiabNet with 1000 associated SNPs, run: 

```bash
python3 training.py configs/simple-training.toml
```

### Full training

To train DiabNet with different datasets (1000 associated SNPs, 1000 non-associated SNPs, 1000 random SNPs, shuffled features, family exclusion), run:

```bash
python3 training.py configs/full-training.toml
```

## Data analysis

Data analysis is performed through a collection of Jupyter Notebooks in the `analysis` directory, that are:

- `01-training-results-analysis.ipynb`
- `...`

First of all, the poetry virtual environment must be manually added to the IPython to be available on Jupyter.

```bash
python -m ipykernel install --user --name=.venv
```

Afterwards, you can run each Jupyter Notebook to visualize DiabNet results with explanations.

```bash
jupyter notebook --notebook-dir analysis
```

However, you may prefer to run all analyzes via bash, that will produce the same images of the Jupyter Notebooks. So run:

```bash
# (NOT WORKING)
python3 analysis/analysis.py
```
