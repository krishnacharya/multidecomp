# Oracle Efficient Algorithms for Groupwise Regret

Code for our ICLR 2024 paper: https://arxiv.org/abs/2310.04652

## Abstract:
We study the problem of online prediction, in which at each time step $t \in {1,2, \cdots T}$, an individual $x_t$ arrives, whose label we must predict. Each individual is associated with various groups, defined based on their features such as age, sex, race etc., which may intersect. Our goal is to make predictions that have regret guarantees not just overall but also simultaneously on each sub-sequence comprised of the members of any single group. Previous work such as [Blum & Lykouris][1] and [Lee et al][2] provide attractive regret guarantees for these problems; however, these are computationally intractable on large model classes (e.g., the set of all linear models, as used in linear regression). We show that a simple modification of the sleeping experts technique of [Blum & Lykouris][1] yields an efficient reduction to the well-understood problem of obtaining diminishing external regret absent group considerations. Our approach gives similar regret guarantees compared to [Blum & Lykouris][1]; however, we run in time linear in the number of groups, and are oracle-efficient in the hypothesis class. This in particular implies that our algorithm is efficient whenever the number of groups is polynomially bounded and the external-regret problem can be solved efficiently, an improvement on [Blum & Lykouris][1]'s stronger condition that the model class must be small. Our approach can handle online linear regression and online combinatorial optimization problems like online shortest paths. Beyond providing theoretical regret bounds, we evaluate this algorithm with an extensive set of experiments on synthetic data and on two real data sets --- Medical costs and the Adult income dataset, both instantiated with intersecting groups defined in terms of race, sex, and other demographic characteristics. We find that uniformly across groups, our algorithm gives substantial error improvements compared to running a standard online linear regression algorithm with no groupwise regret guarantees.

[1] [Blum & Lykouris] https://arxiv.org/abs/1909.08375

[2] [Lee et al]

---

## Code structure:
- Setup: Use the provided conda environment `greg_env.yml` to install all dependencies
- The two algorithms coded from scratch are in the `bilevel` folder, they are `ORidge.py` and `Anh.py`
- Both the real datasets are in the `data` folder
- `jupyter_notebooks` contain `.ipynb` notebooks used to call the above algorithms
- `plots` and `tables` folder contain plots and tables used in the paper
- The `10` random seeds used for shuffling the data are in `bilevel.Groupwise_seedruns.py`, see its constructor they are `[473, 503, 623, 550, 692, 989, 617, 458, 301, 205]`, The random numpy seed we fix for the synthetic data generation ` = 21` and is set explicitly in `groupwise_synthetic.ipynb`
