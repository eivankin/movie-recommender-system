# Movie Recommender System | PMLDL Assignment 2
> Student: Evgenij Ivankin
> 
> E-mail: e.ivankin@innopolis.university
> 
> Group: B21-DS-01

See [task description](task_description.md) for more details about the assignment.

## Reproducing the experiment results
### Training
```shell
python -m src.train models/model.pickle 1 --epochs=30 --plot
```
See `python -m src.train --help` for more details about command parameters. 

### Evaluation
```shell
python -m benchmark.evaluate models/model.pickle 1
```

See `python -m benchmark.evaluate --help` for more details about command parameters.