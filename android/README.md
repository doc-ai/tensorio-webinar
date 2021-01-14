# Tensor/IO Webinar Android Example App

## About

An Android application with the prebuilt model from the Colab notebook already included. Demonstrates how to run the model in an Android application.

## Installation

Open up the android folder in Android Studio, let gradle sync do its magic, choose the *app* configuration and an emulator or hardwire device, and press play!

Code showing how to load and train the model with a batch of image data is found in the *MainActivity*.

## Training the Model

Most of the work of training a model involves loading the image data and preparing a batch for training.

```kotlin
try {
    // Prepare Model

    val bundle = bundleForFile("keras-cifar10-mobilenet-estimator-train.tiobundle")
    val model = bundle!!.newModel() as TrainableModel

    // Prepare Images and Labels

    val stream1 = applicationContext.assets.open("images/cifar-car.jpg")
    val bitmap1 = BitmapFactory.decodeStream(stream1)
    val label1 = numericLabel("automobile")

    val stream2 = applicationContext.assets.open("images/cifar-bird.jpg")
    val bitmap2 = BitmapFactory.decodeStream(stream2)
    val label2 = numericLabel("bird")

    // Prepare Batch

    val item1 = Batch.Item()
    item1.put("image", bitmap1)
    item1.put("label", label1)

    val item2 = Batch.Item()
    item2.put("image", bitmap2)
    item2.put("label", label2)

    val batch = Batch(arrayOf("image", "label"))
    batch.add(item1)
    batch.add(item2)

    // Train Model

    val NUM_EPOCHS = 4
    var losses = floatArrayOf().toMutableList()

    for (epoch in 0 until NUM_EPOCHS) {
        val output = model.trainOn(batch)
        val loss = output.get("total_loss") as FloatArray
        losses.add(loss[0])
    }

    Log.d("Tensor/IO Webinar Demo", "Losses: $losses")

} catch (e: ModelBundle.ModelBundleException) {
    print(e)
} catch (e: Model.ModelException) {
    print(e)
} catch (e: IOException) {
    print(e)
}
```