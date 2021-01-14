# Tensor/IO Webinar iOS Example App

## About

An iOS application with the prebuilt model from the Colab notebook already included. Demonstrates how to run the model in an iOS application.

## Installation

If you don't already have it, install [cocoapods](https://cocoapods.org/). 

Run `pod install` from the iOS root directory then fire up *TensorIOWebinarDemo.xcworkspace* in Xcode, choose a simulator or hardware device, and press play!

Code showing how to load and train the model with a batch of image data is found in *ViewController.swift*.

## Training the Model

Most of the work of training a model involves loading the image data and preparing a batch for training. There is some excessive type casting required to go back and forth between Swift and the Obj-C Tensor/IO implementation.

```swift
// Load the model
                
let bundle = self.bundle(named: "keras-cifar10-mobilenet-estimator-train")!
let model = self.model(with: bundle) as! TIOTrainableModel
    
// Prepare the images and labels
    
let carImage = self.pixelBuffer(for: "cifar-car.jpg")
let carLabel = self.numericLabel(for: "automobile") as NSNumber
    
let birdImage = self.pixelBuffer(for: "cifar-bird.jpg")
let birdLabel = self.numericLabel(for: "bird") as NSNumber
    
// Prepare the batch
// Note that these keys are taken from the model.json and match the keys provided by the serving input receiver fn
    
let batch = TIOBatch(keys: ["image", "label"])
    
batch.addItem([
    "image": carImage,
    "label": carLabel
])
    
batch.addItem([
    "image": birdImage,
    "label": birdLabel
])
    
// Train
    
let NUM_EPOCHS = 4
var losses: [Float] = []
        
for /*epoch*/ _ in 0..<NUM_EPOCHS {
    var error: NSError?
    let output = model.train(batch, error: &error)
    let loss = (output as! NSDictionary)["total_loss"] as! Float
    
    losses.append(loss)
}
    
print("Losses: \(losses)")
```