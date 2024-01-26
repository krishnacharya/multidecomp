0. Both the real dataset are in the `data` folder
1. The two algorithms coded from scratch are in the `bilevel` folder, they are `ORidge.py` and `Anh.py`
2. `jupyter_notebooks` contain `.ipynb` notebooks used to call the above algorithms
3. `plots` and `tables` folder contain plots and tables used in the paper
4. The `10` random seeds used for shuffling the data are in `bilevel.Groupwise_seedruns.py`, see its constructor they are `[473, 503, 623, 550, 692, 989, 617, 458, 301, 205]`
5. The random numpy seed we fix for the synthetic data generation ` = 21` and is set explicitly in `groupwise_synthetic.ipynb`
6. Use the provided conda environment `greg_env.yml` to install all dependencies