# The Attractor Model of Political Competition

This is joint work with [Hans Gersbach](https://mtec.ethz.ch/people/person-detail.hansgersbach.html).
The paper will be available online soon.
Note that visualization look different as my personal style sheet is missing in the repository.

## Abstract

We introduce an attractor model of political competition in which the distribution of voter's ideal points is endogenous.
Two candidates for office select policies they want to pursue in office.
They also select a range of voters they attempt to persuade from the merits of their policies.
Finally, the opinion formation process where each voter is influenced by neighboring opinions determines the distribution of ideal policies.
Our main insight in this paper is that despite of the endogeneity of the distribution of voters' ideal points, candidates adopt policy positions equal to the median voters' ideal point before the communication processes take place.
This result is called the median attractor result. 
We also show that candidates choose smaller influence ranges to attract voter groups as such strategies trigger the most favorable shift of the voter's ideal distribution towards the candidates' policy positions.

## Content

This repository contains code to vizualize the mode in an interactive way and reproduce the results and figures of the project.

### Interactive programe

These programs end on `-interactive.py` and have to be run within `ipython`.

### Reproduce results

All files that end on `-calc.py` or `-plot.py`.

## Run code in a virtual invironment

First setup and active the environment for exmaple with `pip`
```{bash}
python3.8 -m venv ./venv

source ./venv/bin/activate
```

Then install all necessary packages from the `requirements.txt` file
```{bash}
python -m pip install -r requirements.txt 
```

Now you can run the other files.
The `interactive-stationary_solution` file has to be executed within iPython.
