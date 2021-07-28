# Fictitious_classes_ZSL2021

Code accompanying the paper Using Fictitious Class Representations to Boost Discriminative Zero-Shot Learners.
<br> authors:
<br>   Mohammed Dabbah , Ran El-Yaniv
<br>   Department of Computer Science
  Technion – Israel Institute of Technology
<br>   Haifa, Israel 
<br>   mdabbah@campus.technion.ac.il, rani@cs.technion.ac.il

## Instructions

### prerequesits
To install all the dependency packages, please run:
```
pip install -r requirements.txt
```

### Data preperation

Please download the dasets from the following links:
<br> CUB: http://www.vision.caltech.edu/visipedia/CUB-200.html
<br> AWA2: https://cvml.ist.ac.at/AwA2/
<br> SUN: https://groups.csail.mit.edu/vision/SUN/hierarchy.html (using the labelme toolbox)
<br> xlsa17: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly (proposed splits 2.0)

and extract them in <b> dataset </b> directory.

After extraction, the files tree should look like this:

```
.
├── AWA2
│   ├── AwA2-base
│   └── AwA2-data
├── CUB
│   ├── images
│   ├── parts
│   ├── ...
│
├── SUN
│   ├── LabelMeToolbox
│   ├── LabelMeToolbox.zip
│   └── needed_images
│		├── a
│		├── b
│		├── ...
│
├──  xlsa17
│    ├── code
│    ├── data
│    ├── ...
│
│ ...
```

You can download the precomputed features from diffrenet archtictures and diffrerent layers used in our experiments from: <insert karpef link>
or you can use the feature extraction script found at `./main_code/feature_extraction.py`
use 
```
python3 ./main_code/feature_extraction.py -h 
```
to see the scripts options and instructions.

### Reproducing experiments
You can use `./main_code/reproduce.py` to reproduce our main results, and `./main_code/check_backbones.py` to reproduce the backbone and layer choice experiments from the supplimentary material.
use the help flag to see the instructions and options for each script.

```
python3 ./main_code/reproduce.py -h 
python3 ./main_code/check_backbones.py -h 
```

Finally, you can use `./main_code/create_tables.py` to create the main results table and the ablation study table.

### Citation
