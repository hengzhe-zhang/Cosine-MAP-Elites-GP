# MAP-Elites with Cosine Similarity for Evolutionary Ensemble Learning
This repository provides a minimal implementation of MAP-Elites-based Evolutionary Ensemble Learning methods using Genetic Programming for regression tasks. The code uses cosine similarity-based MAP-Elites to create a grid of high-quality and diverse models to form an ensemble model.

## Usage
- 📌 Clone the repository and run main.py to use the code.

- 📊 The code loads and standardizes the Diabetes dataset for training and testing.

- 🧬 Genetic Programming is used to evolve a population of models evaluated by mean squared error on the training data.

- 🔮 Ensemble prediction is performed on the test data using individuals from the population and Hall of Fame, with mean squared error printed.

- 🏆 Three selection methods are compared, with MAP-Elites using Random selection to select high-quality and diverse individuals.

## Results
- 🚀 MAP-Elites with cosine similarity-based PCA and reference points for evolutionary ensemble learning performs well on regression tasks.
- 💪 It outperforms traditional selection operators in terms of lower mean squared error on test data for both population and Hall of Fame ensembles.
- 🌟 Quality-diversity optimization techniques like MAP-Elites can be effectively applied to ensemble learning.

## Dependencies

The code is written in Python and requires the following dependencies:
- 🐍 DEAP
- 🔢 NumPy
- 🧬 scikit-learn


## Reference

The code is based on the paper "MAP-Elites with Cosine-Similarity for Evolutionary Ensemble Learning". Please cite our paper if you find it helpful!

```bibtex
@inproceedings{zhang2023map,
  title={MAP-Elites with Cosine-Similarity for Evolutionary Ensemble Learning},
  author={Zhang, Hengzhe and Chen, Qi and Tonda, Alberto and Xue, Bing and Banzhaf, Wolfgang and Zhang, Mengjie},
  booktitle={Genetic Programming: 26th European Conference, EuroGP 2023, Held as Part of EvoStar 2023, Brno, Czech Republic, April 12--14, 2023, Proceedings},
  pages={84--100},
  year={2023},
  organization={Springer}
}
```

In Figure 8 of the paper, the term "negative cosine similarity" should be revised to "negative cosine distance" in accordance with [the definition of pairwise distance in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances). Apologies for any confusion caused.
