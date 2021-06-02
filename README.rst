**Group B:** Paula Vogg, Danya Li, Felix Hoppe

Milestone 3: Multimodal Predictions & TrajNet++ Challenge
=========================================================

In this third milestone, we trained a **Social Generative Adversial Neetwork (SGAN)** model using the TrajNet++ benchmark. Trying to improve on these results, we thought of another way to forecast pedestrian trajecotries. `This paper <https://openaccess.thecvf.com/content/ACCV2020/papers/Dendorfer_Goal-GAN_Multimodal_Trajectory_Prediction_Based_on_Goal_Position_Estimation_ACCV_2020_paper.pdf>`_ discussing goal-GAN has been a big inspiration for us. Both, our goal-SGAN and the SGAN baseline model, have been evaluated in `the AICrowd challenge <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge>`_. 

Generative Models
-----------------

Generative Models can generate multimodal output, useful to capture the diverse possibilities of future trajectories. The basic idea is that given a past trajectory, there exist multiple possibilities for plausible future trajectories. In the method we want to apply, we use a two-stage process: In a first step we predict the final destination of each actor in a given scene (called goal). Than, in a second step, the goal coordinates are used to predict possible trajectories leading to these goals. Doing so, we hope to achieve good scores for the FDE without increasing the collision rate. 
The overall idea has been visualized by Dendorfer et al.: 

.. raw:: html

    <img src="trained_models/Milestone3/figures/Goal_GAN_dendorfer.png" width="600px">

The training is done seperatly for the goal model and the trajectory (SGAN) model. The goal model uses the the observed trajectories as input, while it's output is compared against the true final coordinates of each actor. The SGAN model is using the observed trajectories and the goal coordiantes as inputs, and returns the coorinates of the primary actor trajectory (during prediction).

Goal Model
----------

The goalModel consists of 2 LSTM layers + 1 lienar layer. For each observed trajectory, we want the goal model to predict multiple possible goals. In order to encourage diversity between the different modes, we used L2-norm-variety-loss during training. 

Two sample situations are shown below:

.. raw:: html

    <img src="trained_models/Milestone3/figures/goal_pred1.png" width="600px">
    

.. raw:: html

    <img src="trained_models/Milestone3/figures/goal_pred2.png" width="600px">

Goal Trainer
------------

To train the goal model, we created a GoalsTrainer class. All code related to training and testing can be found in this file.

SGAN model
----------

In order to use the goal model introduced above, we implemented some changes on the original SGAN model and the corresponding trainer class from the trajnet++ baseline. We decided to only use single-mode SGAN, in order to keep computational complexity during training at a reasonable level. 

Results
--------
    - SGAN single mode, multi mode (k=3) 
    - Goal-GAN












Milestone 2: Implementing Social Contrastive Learning
=====================================================

In this second milestone, we implemented **Social NCE** based on `this paper <https://arxiv.org/pdf/2012.11717.pdf>`_. We implemented both, spatial and event sampling, trained multiple models and finally tested and evaluated them in `this AICrowd challenge <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge>`_. 

Pipeline
--------

.. raw:: html

    <img src="trained_models/Milestone2/figures/pipeline.png" width="600px">
    


Social Contrastive Learning
--------

Contrastive learning used with negative data augmentation is said to increase the robustness of the neural motion models. The basic idea of contrastive learning is to use a simple similarity measure between our learned embeddings (truth, positive and negative samples) to approximate the preferred neighborhood relationships. The special samlping strategy in *social* contrastive learning is based on our domain knowledge of socially unfavorable events in the multi-agent context, which observed that it is typically forbidden or uncomfortable for multiple agents to visit the same or adjacent places simultaneously. As shown in the image we want to predict the trajectory of the primary pedestrian and use the position of the neighbours and their sourroundings to tell the model which future steps are not a good choice. 

.. raw:: html

    <img src="trained_models/Milestone2/figures/SCL_negative_data_augmentation.png" width="400px">



Contrastive Sampling
--------

The single-frame sampling algorithm (samples locations at a specific time of the future) follows the following steps: 

* **Positive samples:** Given a fixed future horizon, we select the corresponding sample from the ground truth of primary agent and add some noise to it. 
* **Negative samples:** Given a fixed future horizon, we select the corresponding sample from the ground truth of neighboring agents with local displacement and add some noise to them. It is worth mentioning that treating negative samples was more challenging, as the number of neighbors (agents other than the primary agent) might vary from scene to scene. In order to have the same tensor size for all scenes, we filled up scenes (with few neighbors and NaN's) with random samples from neighbors of the same scene. This shouldn't change the overall outcome, as we are randomly assigning a higher weight to a neighbor.

The multi-frame sampling is very silimar. The difference is that samples are spatial-temporal events at various time steps of the future. 

In the figure below we plotted the raw trajectories of the pedestrians as well as our samples at the desired time horizon. Nine negative samples per neighbour in red dot are shown, while the green point shows the positive sample. Remember for each scene one primary pedestrian and several neighbours are chosen. The trajectroy of the primary gives the postive sample and the trajectories of the neigbours give the negative samples. In addition to that, the observed and the future trajectory as well as the horizon (here horizon = 4) are shown.

.. raw:: html

    <img src="trained_models/Milestone2/figures/data_sampling_synth_data.jpeg" width="400px">


Having created our samples, we performed the following steps for spatial NCE:

* Lower dimensional embedding of observations (past trajectories) and positive / negative samples
* Normalization of all lower dimensional embeddings
* Computation of pairwise similarity
* Computation of NCE Loss


Training
--------
While training, once our code performed without error, we investigate different hyperparameters:

* contrastive weight (relative weight of NCE loss compared to the normal loss)
* contrastive temperature (for down or upscaling of similarity)
* horizon 

In general we trained the models on both data sets (real and synthetic data). The following combinations were trained: 

* weight = 1, temperature = 0.1, horizon = 4, skip (synth), replace (real)
* weight = 2, temperature = 0.1, horizon = 4, replace
* weight = 1, temperature = 0.2, horizon = 4, skip (synth), replace (real)
* weight = 1, temperature = 0.1, horizon = 8, replace
* weight = 1, temperature = 0.1, horizon = 12, replace

Note that in the first place we used the skipping technique (skipping the scenes with any NaN's) to deal with NaN values. This didn't work for real data due to the high amount of NaN values. Therefore we changed to the replacement technique (replacement of NaN's with random samples from other neighbors. If there are no neighbors or all existing neighbors have only NaN values, we replace them by (-10,-10)). The models trained using skipping were trained on synthetic data and we think the performance for synthetic data uing skipping or replacing is similar (as in general we only had very few NaN's here).


Evaluation & Results
--------------------

Learning Curves of real data set
+++++++++++++++

.. raw:: html

    <img src="trained_models/Milestone2/figures/real_data_learning_curves.png" width="400px">

The above figure shows the learning curves of all 5 models which have been trained on the real data set. The curves look very similar for the first 4 models. The 5th model has been has been pretrained for 25 epochs on synth_data. As we didn't reset the learning rate, it used a much lower learning rate as the other models. Considering the high initial loss, we can make the assumption that training on synth_data does not generalize very well to real_data.

Learning Curves of synthetic data set
+++++++++++++++

.. raw:: html

    <img src="trained_models/Milestone2/figures/synth_data_learning_curves.png" width="400px">

We trained 4 different models on synth_data, although unfortunately only 2 have been trained up to epoch 25 (in order to avoid too extensive computation times). As the use of different parameters effects the way the loss has been computed, we can't draw any conclusions directly from this plot but rather have to consider the evaluation metrics.


Evaluation of synthetic dataset models on five_parallel_synth
++++++++++

.. raw:: html

    <img src="trained_models/Milestone2/figures/synth_data_eval.jpg" width="800px">

The above table is showing the evaluation results from all models trained on *synth data*, and tested on *five_parallel_synth*. The two relevant metrics for AICrowd are FDE (final deplacement error) and Col-I (collision); for both lower is better. On the *five_parallel_synth* data set, all *single* models are giving the same results. The one *multi* model has a higher Col-I score and a lower FDE score. This seems reasonable, given that the model trains on dynamic negative samples and thus will be less cautious about collisions.  

Evaluation from AI crowd
+++++

In the table below, for each model the hyperparameters and the evalution score from AI crowd are given. In Milestone 1 our best model reached FDE = 1.210 and Col-I = 7.220, while now we achieve **FDE = 1.200** and **Col-I = 4.710 with contrastive learning!**

+------------+------------+-------------+----------+-------+-------+
|  weight    | horizon    | temperature | sampling | FDE   | Col-I |
+============+============+=============+==========+=======+=======+ 
| 1          | 4          | 0.1         | single   | 1.200 | 4.710 |
+------------+------------+-------------+----------+-------+-------+ 
| 1          | 4          | 0.2         | single   | 1.210 | 5.310 |
+------------+------------+-------------+----------+-------+-------+ 
| 1          | 8          | 0.1         | single   | 1.250 | 5.190 |
+------------+------------+-------------+----------+-------+-------+ 
| 1          | 12         | 0.1         | single   | 1.220 | 4.470 |
+------------+------------+-------------+----------+-------+-------+ 
| 2          | 4          | 0.1         | single   | 1.200 | 5.310 |
+------------+------------+-------------+----------+-------+-------+ 
| 1          | 4          | 0.1         | multi    | 1.220 | 4.470 |
+------------+------------+-------------+----------+-------+-------+

From our results, we can draw the following conclusions: 

* Social NCE sampling improves results
* augmenting the temperature to 0.2 does not increase the performance.
* augmenting the time horizon does decrease the overall performance of the model, however for h = 12 we find that the collision test actually gets better.
* augmenting contrastive weight form 1 to 2, decreases the model’s performance. 
* and applying the sampling strategy multi increases the FDE and decreases the Col-I.





AICrowd submission
++++++++++++++++++

Our AICrowd submission can be found here: `Link <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge/submissions/138580>`_













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

