Author: Krishna Acharya, Date: 5 July 2023
Follow these steps for reproduction (copy and paste in bash terminal):

1) First run the following to make a conda environment named multigroup with basic python build:
    conda create -n multigroup python

2) Activate the environment:
    conda activate multigroup

install dependencies
    conda install ipykernel pandas matplotlib
    pip install jupyterlab river folktables tqdm

3) Register the kernel for use in the jupyter-lab demo
    ipython kernel install --user --name=multigroup
    
4) View the examples_xyz.ipynb by launching jupyterlab
    jupyter-lab

Set multigroup kernel in the top right, or as prompted on startup

main notebook to open is 3plotting_AnhvsORidge_results.ipynb

In general the pipeline for any statelist we want to run the code for is the following
1example_dataprocessing.ipynb
2examplemain_AnhvsORidge_all.ipynb
3plotting_AnhvsORidge_results.ipynb