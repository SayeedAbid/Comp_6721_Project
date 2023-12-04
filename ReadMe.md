#﻿COMP 6721 Applied Artificial Intelligence
##Table of Contents<br>
1. Introduction
2. Code Execution
   * Recommended modules
   * Installation
   * Procedure
     * Data Cleaning
     * Data Visualization
     * Model Train and Evaluate <br>
## Introduction<br>
This project includes several files, and the files are listed below:
* Python Code
* Dataset
* Project Report
* Read Me
* Originality Form<br>
The Python code folder includes the codes for all the data preprocessing, analysis, and data visualization code. Inside the folder, there are several executable Python scripts. The data_analysis.py file holds the Python script for the data cleaning and augmentation. The visualization.py script holds the codes for the data visualization task. The other files are used for training and testing the models. The procedure for running those files will be explained later in this file.<br>
In the dataset folder, there is a file named dataset.txt, which will explain the sources of the dataset and the link to our finalized dataset. Along with this file, there is a folder named Data, which has a demo dataset where you can find 10 representative images from each class of our dataset. There is another folder named Bias_analysis_data, which is labeled according to the selected bias attributes and can be used for bias analysis.<br>
The Report.pdf file is the main project report, which includes all the implementation details, results, explanations, tables, figures, and references of our project.<br>
The ReadMe file will explain the purpose of each of the deliverables in the submission folder. It will also explain the procedure to run the codes.<br>
The Originality form folder holds all the signed originality contracts from each of the team members.<br>
 
## Code Execution<br>
This section will explain everything about executing the Python scripts used in this project. Follow the steps below to execute the provided codes successfully.<br>
### Recommended Modules<br>
This program has the following system and library requirements:
* Windows/Linux operating system
* Any standard IDE (VS Code, PyCharm etc.)
* Python 3.7 or above
* Pytorch
* NumPy
* Matplotlib
* Scikitlearn
* Seaborn
* OpenCV<br>
### Installation<br>
If you need to install Python in your system, please visit this [site](https://www.python.org/). For installing the standard Python libraries, please visit this [site](https://docs.python.org/3/library/index.html).<br>
### Procedure<br>
#### Data Cleaning<br>
To run the Python scripts of the data cleaning and analysis part, please open the file named data_analysis.py in your IDE. Then, you will see that the code is divided into different parts for different operations. Keep the useful imports part and the part of the code related to your desired operation, and comment on the rest of the code. In this way, for each operation you want to perform, just keep the imports and corresponding code uncommented and comment out everything else, and then just run the code. Please make sure to modify the data source according to your specified folder where mentioned.<br>
#### Data Visualization<br>
For the Python script on data visualization, please just open the visualization.py script in your IDE, then modify the data source and run the file. You will get the desired results.
#### Model Train and Evaluate
To train the CNN network and Evaluate, simply run the given .py files. The model_train.py file is our main model, which gives us the best results. The variant1.py and variant2.py are the two variants of our model. By running the files, it will first train the network on the specified model, then generate the confusion matrix and the Macro and Micro scores. The "Scores" folder has all our trained scores for the project. If you want to run the model on k-fold cross-validation, then you will have to run the k_fold_train.py file. Please make sure to specify the correct dataset for the file. To do the bias analysis, you can use the test.py file and the "Bias Test Data." <br>
You can also use this test file to evaluate other models as well.
