# Visualization selection

[![Launch in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KrishnaswamyLab/visualization_selection/master?filepath=Selecting_the_right_tool_for_the_job.ipynb)

This is the repository used to generate and host the following paper accepted to 2nd Workshop on Visualization for AI Explainability ([VisxAI](visxai.io)):  

**Selecting the right tool for the job: understanding drawbacks and biases in visualization methods**  
[Daniel B Burkhardt\*](mailto:daniel.burkhardt@yale.edu), [Scott A Gigante\*](mailto:scott.gigante@yale.edu), [Smita Krishnaswamy](mailto:smita.krishnaswamy@yale.edu).    
Yale University, New Haven, CT.  

[Click here to view the interactive notebook on Binder](https://mybinder.org/v2/gh/KrishnaswamyLab/visualization_selection/master?filepath=Selecting_the_right_tool_for_the_job.ipynb)

If you have any questions or want to get in touch, feel free to add an Issue or send us an email by clicking on one of the author names above!

# About the repository

This repository is broken into several components.

### blog_tools

Most of the interactive components of the notebook are powered by the `blog_tools` package. In this package are several modules for generating data, convenience wrappers for running each embedding tool, and interactive IPython and Plotly widgets that facilitate exploring the `Selecting_the_right_tool_for_the_job.ipynb` notebook. The `setup.py` and `requirements.txt` files in the root of this repository are for the `blog_tools` package. We wrote these tools specifically for this workshop paper, so you may not find them immediately applicable to a new project. However, we hope they serve as an example of how Plotly and Jupyter can be used to create an interactive explorable notebook.

### data

This directory holds some data used in generating the figures in our article, and when you run some of the helper notebooks, data will be downloaded into this folder.

### img, md

Source directories for images and markdown text that appears in the article respectively.

### notebooks

This folder contain Jupyter notebooks that were used to generate the figures used in the final article. The notebooks names refer to the dataset that is analyzed within that notebook. More details about the methods can be found within the notebooks.

### scripts

This folder holds Python scripts that were used to generate the images for the parameter sensitivity analysis.

### Selecting_the_right_tool_for_the_job.ipynb

This is the notebook that is opened in the binder link above and contains all of the workshop paper accepted to VisxAI.
