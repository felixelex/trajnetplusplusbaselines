**Group B:** Paula Vogg, Danya Li, Felix Hoppe

Milestone 2: Implementing Social Contrastive Learning
=====================================================

In this second milestone, we implemented **Social NCE** based on `this paper <https://arxiv.org/pdf/2012.11717.pdf>`_. We implemented both, spatial and event sampling, trained multiple models and finally tested and evaluated them in `this AICrowd challenge <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge>`_. 

Social Contrastive Learning
--------

Contrastive learning used with negative data augmentation has been said to increase the robustness of forecasting models. In the data of pedestrian trajecotries, we define some as hsitorical data and use the remaining data to create positive and negative samples. For that we choose a primary pedestrian and some neighbours for each scene. As shown in the image we want to predict the trajectory of the primary pedestrian and use the position of the neighbours and their sourroundings to tell the model which future steps are not a good choice. 

.. raw:: html

    <img src="trained_models/Milestone2/figures/SCL_negative_data_augmentation.png" width="400px">


Sampling
--------

Spatial sampling
++++++++++++++++

The spatial sampling algorithm follows the following steps: 

* **Positive samples:** Given a fixed horizon, we select the corresponding sample from the ground truth and add some noise to it. 
* **Negative samples:** Treating negative samples was more challenging, as the number of neighbors (agents other than the primary agent) might vary from scene to scene. In order to have the same tensor size for all scenes, we filled scenes with few neighbors and NaN's up with random samples from neighbors of the same scene. This shouldn't change the overall outcome, as we are randomly assigning a higher weight to a neighbor.

In the figure below we plotted the trajectories of the pedestrians and added the sampling at the time horizon in the plot. In red there are the nine negative sample per neighbour and the green point shows the positive sample. Remember for each scene one primary pedestrian and several neighbours are chosen. The trajectroy of the primary gives the postive sample and the trajectories of the neigbours give the negative samples. In addition to that, the observed and the future trajectory as well as the horizon (here horizon = 4) are shown.

.. raw:: html

    <img src="trained_models/Milestone2/figures/data_sampling_synth_data.jpeg" width="400px">


Having created our samples, we performed the following steps for spatial NCE:

* Lower dimension embedding of observations (past trajectories) and positive / negative samples
* Normalization of all lower dimensional embeddings
* Computation of NCE Loss


Training
--------



Evaluation & Results
--------------------







Milestone 1: Getting Started
============================

In this readme, we present the results of our insightful struggle through the first part of the DLAV project. 

Ressources
----------

Tutorial: `Link <https://thedebugger811.github.io/posts/2021/04/milestone_1/>`_  

Trajnet++ Baselines Repository: `Link <https://github.com/vita-epfl/trajnetplusplusbaselines/>`_  

Approach
========

Here is what we did: 

* In the first step, we went through the setup process on our local machine. We trained some simple models (vanilla LSTM, directional LSTM) on a small data set (five_parallel_synth_split). Furthermore, we evaluated these models and plotted statistics and predictions.
* In the second step, we went once more through the setup process, but this time on the EPFL SCITAS server. After becomming Masters of the command line, we managed to run the same training jobs as we did previously on our local machine - showing that everything works just fine.
* Next, we trained a number of models on different data sets. We used both our local machines as well as SCITAS to run these trainings. A list of all trained models can be found in the *Training models* section.
* In the next step, we evaluated different models. While using the extensive scoring of *Trajnet++*, we also plotted the model predictions in different situations. For more details, see *Evaluation and Results*.
* Finally, we picked our best performing model, and uploaded an submission to `AICrowd <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge>`_.

Our trained models and result visualizations are placed at *./trained_models*.

Training Models
=======

Motivated as we are, we trained the following models (organized by training data set):

*five_parallel_synth*

- vanilla
- directional

*synth_data*

- vanilla 
- directional (with goals)
- attentionmlp (with goals)

*real_data*
 
- attentionmlp (canceled, took too long)

*real_data_noCFF* (subset of real_data)

- vanilla
- directional
- attentionmlp



Training statistics
-------------------

.. raw:: html

    <img src="trained_models/figures/lstm_attentionmlp_None.pkl.log.epoch-loss.png" width="600px">

.. raw:: html

    <img src="trained_models/figures/lstm_attentionmlp_None.pkl.log.train.png" width="600px">

Considering the two plots above, we can note several things:

- The loss decreases for all models. This implies that all models are able to learn from the data.
- There is a jump in the performance improvement after epoch 10. This coincides with the scheduled decrease of the learning rate after epoch 10. The second learning rate decrease after epoch 20 has no major effect.
- The standard deviation of the loss function remains quite large throughout the training.
- No matter which dataset is used, models considering interaction between pedestrains always have lower loss than the vanilla ones. 



Evaluation and Results
======================

All models have been tested on the *five_parallel_synth/test_private* data.

Overall results analysis
------------------------

Models trained on **five_parallel_synth (fps)** data

.. figure:: trained_models/five_parallel_synth/Results_cropped.png
  :width: 400

Models trained on **synth_data (sd)** data

.. figure:: trained_models/synth_data/Results_cropped.png
  :width: 400

Models trained on **real_data_noCFF (rd)** data

.. figure:: trained_models/real_data_noCFF/Results_cropped.png
  :width: 400 

**Metrics:** 

Average Displacement Error (**ADE**): Average L2 distance between the ground truth and prediction of the primary pedestrian over all predicted time steps. Lower is better.

Final Displacement Error (**FDE**): The L2 distance between the final ground truth coordinates and the final prediction coordinates of the primary pedestrian. Lower is better

Prediction Collision (**Col-I**): Calculates the percentage of collisions of primary pedestrian with neighbouring pedestrians in the scene. The model prediction of neighbouring pedestrians is used to check the occurrence of collisions. Lower is better.

Ground Truth Collision (**Col-II**): Calculates the percentage of collisions of primary pedestrian with neighbouring pedestrians in the scene. The ground truth of neighbouring pedestrians is used to check the occurrence of collisions. Lower is better.

**Interpretation of results:**

In the comparison of the two different kinds of models (with or without interaction encoder), the errors for predictions using the vanilla model are much higher compared to using a directional model. This makes sense, because the vanilla model does not take into account the interaction between pedestrians, whereas the model using a directional interaction encoder considers the interaction between pedestrians. Therefore it is logical that for all three data sets, we have lower errors for the model using a interaction encoder. These interaction encoders were either 'directional' or on the training with real data we tested also the 'attention MLP' encoder. 

Having a look at the difference of using a directional or an attention MLP encoder in the real dataset we can see that the performance is very similar. Although training took a lot longer for the attention MLP model. 

Comparing the Col-I and the Col-II errors, we observe a much higher error for the colision testing Col-II in the case of the interaction encoder models. Col-II is looking at the collision of the predicted way of pedestrians with the groundtruth, whereas the Col-I takes into account only the prediction within the model. Therefore it makes sense that there are more errors when comparing to the groundtruth and the low error of Col-I means that our model still has a good performance because it understood that it needs to avoid pedestrian's collision. For the vanilla model both errors Col-I and Col-II are high, this means that the model is really bad in avoiding collisions, which makes sense because it does not take into account interactions. 



Predictions visualization 
-------------------------
       
Below, predictions of trained models in 2 different situations are shown:

SCENE ID: 43906

*five_parallel_synth*

.. raw:: html

    <img src="trained_models/figures/fps-visualize.scene43906.png" width="400px">

*real_data_noCFF*

.. raw:: html

    <img src="trained_models/figures/no-visualize.scene43906.png" width="400px">

*synth_data*

.. raw:: html

    <img src="trained_models/figures/sd-visualize.scene43906.png" width="400px">
    
    
SCENE ID: 46845

*five_parallel_synth*

.. raw:: html

   <img src="trained_models/figures/fps-visualize.scene46845.png" width="400px">

*real_data_noCFF*

.. raw:: html

   <img src="trained_models/figures/no-visualize.scene46845.png" width="400px">

*synth_data*

.. raw:: html

   <img src="trained_models/figures/sd-visualize.scene46845.png" width="400px">
   
   
SCENE ID: 48031

*five_parallel_synth*

.. raw:: html

   <img src="trained_models/figures/fps-visualize.scene48031.png" width="400px">

*real_data_noCFF*

.. raw:: html

   <img src="trained_models/figures/rd_no-visualize.scene48031.png" width="400px">

*synth_data*

.. raw:: html

   <img src="trained_models/figures/sd-visualize.scene48031.png" width="400px">


**Interpretation of results:**

For the visualisation we took the trained models and tested them on *five_parallel_synth* dataset which has all available goal files. This might explain why those models trained on other datasets (*synth_data* and *real_data*) perform not as good as the models trained on *five_parallel_synth* dataset. This can also be seen from *Overall result analysis* above. Furthermore we can observe that the predictions made by a D-Grid model (with interaction encoder) are anticipitating better the actual trajectory. In the case of the model trained on the *real_data* it is possible that the lack of goal information (we do not know where pedestrians want to go) makes it more difficult to do the proper predictions. 

AICrowd submission
==================

Our AICrowd submission can be found here: `Link <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge/submissions/132459>`_





Reference
=========

The used Trajnet++ Baseline code has been developed by

.. code-block::

    @article{Kothari2020HumanTF,
      title={Human Trajectory Forecasting in Crowds: A Deep Learning Perspective},
      author={Parth Kothari and S. Kreiss and Alexandre Alahi},
      journal={ArXiv},
      year={2020},
      volume={abs/2007.03639}
    }

