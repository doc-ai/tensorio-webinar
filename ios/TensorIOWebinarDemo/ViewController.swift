//
//  ViewController.swift
//  TensorIOWebinarDemo
//
//  Created by Philip Dow on 1/14/21.
//

import UIKit
import TensorIO

class ViewController: UIViewController {

    @IBOutlet var imageView1: UIImageView!
    @IBOutlet var imageView2: UIImageView!
    @IBOutlet var epochTextField: UITextField!
    @IBOutlet var lossesTextView: UITextView!
    
    // Model Loading Utilities
    
    func bundle(named name: String) -> TIOModelBundle? {
        guard let path = Bundle.main.url(forResource: name, withExtension: "tiobundle", subdirectory: "models")?.path else {
            print("Unable to find the model with name \(name)")
            return nil
        }
        
        return TIOModelBundle(path: path)
    }
    
    func model(with bundle: TIOModelBundle) -> TIOModel? {
        guard let model = bundle.newModel() else {
            print("There was a problem instantiating the model from the bundle")
            return nil
        }
        
        guard let _ = try? model.load() else {
            print("There was a problem loading the model")
            return nil
        }
        
        return model
    }
    
    // Image Loading Utilities
    
    func image(named name: String) -> UIImage {
        let filename = URL(string: name)!.deletingPathExtension().path
        let ext = URL(string: name)!.pathExtension
        
        let path = Bundle.main.url(forResource: filename, withExtension: ext, subdirectory: "images")!.path
        let image = UIImage(contentsOfFile: path)!
        
        return image
    }
    
    func pixelBuffer(for name: String) -> TIOPixelBuffer {
        let image = self.image(named: name)
        let pixels = image.pixelBuffer()!
        let value = pixels.takeUnretainedValue() as CVPixelBuffer
        let buffer = TIOPixelBuffer(pixelBuffer: value, orientation: .up)
        
        return buffer
    }
    
    /// CIFAR-10 Labels
    
    let labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]
    
    func numericLabel(for label: String) -> Int {
        return labels.firstIndex(of: label)!
    }
    
    /// API Overrides
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Done Button
        
        let bar = UIToolbar()
        
        bar.items = [
            UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil),
            UIBarButtonItem(barButtonSystemItem: .done, target: self, action: #selector(dismissEpochsKeyboard(sender:)))
        ]
        
        bar.sizeToFit()
        epochTextField.inputAccessoryView = bar
        
        // Image Preview
        
        imageView1.image = UIImage(contentsOfFile: Bundle.main.path(forResource: "cifar-car", ofType: "jpg", inDirectory: "images")!)
        imageView2.image = UIImage(contentsOfFile: Bundle.main.path(forResource: "cifar-bird", ofType: "jpg", inDirectory: "images")!)
    }
    
    @IBAction
    func dismissEpochsKeyboard(sender: UIResponder) {
        epochTextField.resignFirstResponder()
    }
    
    @IBAction
    func trainAndReport(sender: UIResponder) {
        let epochs = Int(epochTextField.text!)!
        let losses = train(epochs: epochs)
        
        lossesTextView.text = losses.description
            .replacingOccurrences(of: "[", with: "")
            .replacingOccurrences(of: "]", with: "")
            .replacingOccurrences(of: ", ", with: "\n")
        
        print("Losses: \(losses)")
    }
    
    /// This is what you're interested in.
    ///
    /// In real life you'd run this off the main thread. In this example the image assets have
    /// already been resized to 96x96 but that is not necessary. Tensor/IO will take care of resizing
    /// images and applying any other transformations you declare in the model.json file.
    
    private func train(epochs: Int) -> [Float] {
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
        
        var losses: [Float] = []
                
        for /*epoch*/ _ in 0..<epochs {
            var error: NSError?
            let output = model.train(batch, error: &error)
            let loss = (output as! NSDictionary)["total_loss"] as! Float
            
            losses.append(loss)
        }
        
        return losses
    }
    
}

