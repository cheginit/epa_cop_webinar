# EPA's CoP Webinar

In this webinar, we will focus on the classification of NHD catchments across CONUS into
three types based on their hydrological characteristics, specifically examining drought
propagation mechanisms based on [Apurv et al., (2017)](http://dx.doi.org/10.1002/2017WR021445).
Utilizing StreamCat data, we will train a machine-learning model and leverage the HyRiver
software stack for efficient data retrieval and processing operations.

## Prerequisites

We use the following software stack:

* Python 3.10
* [HyRiver](https://docs.hyriver.io): For data retrieval and processing
* [PyTorch Tabular](https://pytorch-tabular.readthedocs.io): For training a machine learning model and making predictions over the entire CONUS

## Installation

Note that PyTorch Tabular is not available on `conda-forge` yet, so we will install it from
pip. First, create a new conda/mamba environment:

```bash
conda create -n cop-webinar python=3.10
conda activate cop-webinar
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```
