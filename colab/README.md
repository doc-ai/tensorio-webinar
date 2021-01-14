# Tensor/IO Webinar Colab

## About

The notebook in this directory can be run as-is in [Google Colab](https://colab.research.google.com). It will build the model that is used in the iOS and Android demo apps and download its saved model contents to your filesystem.

Open the notebook in Colab directly from GitHub by choosing File > Open Notebook, selecting the GitHub tab, entering this repository's URL (https://github.com/doc-ai/tensorio-webinar) and then choosing the colab/tensorio_demo.ipynb notebook that Colab finds.

## Packaging the Model

You must package the resulting model in a [Tensor/IO model bundle](https://github.com/doc-ai/tensorio-ios/wiki/Packaging-Models) in order to be able to use it on device. Tensor/IO model bundles are just folders that contain the model and a *model.json* file describing the model. The *model.json* files needed for the prediction and training exports of this model are provided in this repository as *model.predict.json* and *model.train.json*. Package those json files alongside your model export and rename them to *model.json*.

For example the inference model bundle will look like:

```
keras-cifar10-mobilenet-estimator-predict.tiobundle
├── model.json
└── predict
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

And the training model bundle will look like:

```
keras-cifar10-mobilenet-estimator-train.tiobundle
├── model.json
└── train
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

Make sure you use the model exported for inference with the prediction *model.json* and the model exported for training with the training *model.json*. The signature definitions of the two models are different!

Note that the names of the directories that contain the exported model, *predict* and *train* respectively, correspond to value of the **model.file** field in the *model.json*. You can use whatever names you want as long as they match.