# LinearModel
1.Project Structure:
    ./data/:
        train.conll: the training data.
        dev.conll: the develop data.
    ./LM:
        LM.py: the original code file for LinearModel.
        LM_v1.py: the code file for LinearModel with the averaged method.
    ./LM_2d:
        LM_v2.py: the code file for LinearModel with the averaged method using the 2d array.
    ./README.md:
        Something to introduce the project
    ./result.txt:
        The programs' result on mu computer.

2.How to run the LM?
    We have three code files in total in the project and we can run them all.
    To run these project, you should prepare the following things:
        python2.7
        move data files in the same folder with the code files
    2.1: run:
         python ./LM.py
         python ./LM_v1.py
         python ./LM_v2.py

You can run all the projects and compare them with the precision and executing time.
