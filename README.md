# Projet "Apprentissage et Génération par Échantillonnage de Probabilités"


## Introduction

This project is part of the "Learning and Sampling Generation by Probabilistic Sampling" course taught by Stéphane Mallat at the Collège de France. We are currently enrolled in the Master of Computer Vision and Learning (MVA) program, where this project serves as a practical application of the concepts covered in the course.

## Project Description

The main objective of this project is to develop a system for segmenting defects in conduit images. Defect segmentation is a critical task in various industries, including manufacturing and infrastructure maintenance. By precisely identifying and delineating defects in images, we can facilitate timely repairs and preventive maintenance, thereby improving safety and operational efficiency. This project is part of a data challenge organized by ENS Paris, you can check [ENS data challenge](https://challengedata.ens.fr) for further information.

## Methodology

We have developped a U-Net architecture-like model. After many experiments, the hyperparameters and the training method has been adapted to perform an accurate segmentation of images of corroded wells. Finally, we succeed in obtaining a high IoU score (0.66, ranked first among the students of our Master) on the test set. 

## Usage

Choose your model and the hyperparameters and then run:

    ```
    python3 train.py --wandb --wandb_entity yourid_wb --batch-size 64 --num-epochs 200 --model-name cat_unet --criterion iou -lr 1e-4  --pretrained --experiment_name new_exp
    ```
## Collaborators

- Hippolyte Pilchen (MVA, ENS Paris-Saclay)
- Lucas Gascon (MVA, ENS Paris-Saclay)

We can be reached at the following email address: forename.name@polytechnique.edu


## Acknowledgments

We would like to express our gratitude to Stéphane Mallat for his insightful lectures and guidance throughout the course. Additionally, we acknowledge the ENS and SLB for designing this challenge.
