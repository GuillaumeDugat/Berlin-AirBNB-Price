# Airbnb Berlin Price Prediction

## Description

The project aims to predict the price of a night's accommodation offered on the short-term rental website Airbnb in the city of Berlin. This project is derived from a data challenge proposed by dphi.tech. Details of the data challenge is [available in this link](https://dphi.tech/challenges/data-sprint-47-airbnb-berlin-price-prediction/160/data).



## Structure of the repository

The repository is made up with a main.py file that is used to do all the pipeline: preprocessing, training and evaluation. It also contains a jupyter notebook, named testing.ipynb that was used to determine the parameters of pca, pls, and subset selection method. The folder 'data' contains the dataset split into training.csv and test.csv, the folder preprocessing contains the classical preprocessing (named baseline preprocessing) and the more advanced techniques (pca, pls, forward and backward selection) and the folder models contains every model that we tested.


## Installation

Clone this repository:

```
git clone https://gitlab-student.centralesupelec.fr/2019brillandt/apprauto-airbnb-berlin.git
```

You will need to have python installed, as well as some packages (which can be install easily with pip).

## Usage

To run the project and do the preprocessing, fitting and evaluation, the following command should be executed from the root of the project:

```
python main.py
```

To try different models and preprocessing, you have to open main.py and change directly the file to select the ones you want (the lines that can be modified are clearly indicated).
