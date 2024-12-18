.. DeepLib documentation master file, created by
   sphinx-quickstart on Wed Dec 18 11:50:04 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepLib
=============

DeepLib is a unified PyTorch library for computer vision tasks, focusing on object detection, semantic segmentation, and anomaly detection.

Installation
-----------

Prerequisites
~~~~~~~~~~~~

DeepLib requires PyTorch and torchvision to be installed first. For optimal performance, CUDA 11.8 or above is recommended:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

For other installation options (CPU-only, different CUDA versions), see the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

Installing DeepLib
~~~~~~~~~~~~~~~~

Full Installation (Recommended):

.. code-block:: bash

   pip install deeplib

Core Installation (without logging backends):

.. code-block:: bash

   pip install deeplib[core]

Features
--------

- **Semantic Segmentation Models** (✅ Implemented)
    - UNet
    - DeepLabV3
    - DeepLabV3+

- **Experiment Tracking** (✅ Implemented)
    - TensorBoard
    - MLflow
    - Weights & Biases (W&B)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   examples/index
   api/index

Quick Start
----------

Here's a simple example of training a segmentation model:

.. code-block:: python

   from deeplib.models.segmentation import UNet
   from deeplib.trainers import SegmentationTrainer
   from deeplib.datasets import SegmentationDataset
   from deeplib.loggers import WandbLogger
   from torch.utils.data import DataLoader

   # Initialize model
   model = UNet(num_classes=4)

   # Prepare dataset
   train_dataset = SegmentationDataset(
       root="path/to/data",
       images_dir="images",
       masks_dir="masks",
       num_classes=4,
       split="train"
   )

   # Create dataloader
   train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

   # Configure loss function
   model.configure_loss('dice', {'ignore_index': 255})

   # Initialize logger
   logger = WandbLogger(
       experiment_name="segmentation_experiment",
       project="deeplib-segmentation"
   )

   # Train model
   trainer = SegmentationTrainer(
       model=model,
       train_loader=train_loader,
       monitor_metric='iou',
       logger=logger
   )

   trainer.train(num_epochs=100)

For more examples and detailed usage, check the :doc:`examples/index` section.

Indices and Tables
----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

