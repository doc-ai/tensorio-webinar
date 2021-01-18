package ai.doc.tensorio.webinardemo

import ai.doc.tensorio.core.data.Batch
import ai.doc.tensorio.core.model.Model
import ai.doc.tensorio.core.modelbundle.ModelBundle
import ai.doc.tensorio.core.training.TrainableModel
import ai.doc.tensorio.core.utilities.AndroidAssets
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.IOException

class MainActivity : AppCompatActivity() {

    /** CIFAR-10 Labels */

    private val labels = arrayOf(
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
    )

    /** Tensor/IO takes int[] inputs in Java so we must wrap a single int index in an array */

    private fun numericLabel(name: String): IntArray {
        return intArrayOf(labels.indexOf(name))
    }

    /** Create a model bundle from a file, copying the asset to models */

    @Throws(IOException::class, ModelBundle.ModelBundleException::class)
    private fun bundleForFile(filename: String): ModelBundle? {
        val assetpath = "models/$filename"
        val dir = File(applicationContext.filesDir, "models")
        val file = File(dir, filename)

        if (!dir.exists()) {
            dir.mkdir()
        }

        if (!file.exists()) {
            AndroidAssets.copyAsset(applicationContext, assetpath, file)
        }

        return ModelBundle.bundleWithFile(file)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Show Images

        val imageView1 = findViewById<ImageView>(R.id.image1)
        val imageView2 = findViewById<ImageView>(R.id.image2)

        val stream1 = assets.open("images/cifar-bird.jpg")
        val image1 = BitmapFactory.decodeStream(stream1)

        val stream2 = assets.open("images/cifar-car.jpg")
        val image2 = BitmapFactory.decodeStream(stream2)

        imageView1.setImageBitmap(image1)
        imageView2.setImageBitmap(image2)
    }

    /** Called by the Train button */

    public fun trainAndReport(view: View) {
        val epochs = findViewById<EditText>(R.id.epochsValue).text.toString().toInt()
        val losses = train(epochs)

        findViewById<TextView>(R.id.lossValues).text = losses.toString()
            .replace("[", "")
            .replace("]", "")
            .replace(", ", ",\n")

        Log.d("Tensor/IO Webinar Demo", "Losses: $losses")
    }

    /**
     * This is what you're interested in.
     *
     * In real life you'd run this off the main thread. In this example the image assets have
     * already been resized to 96x96 but that is not necessary. Tensor/IO will take care of resizing
     * images and applying any other transformations you declare in the model.json file.
     *
     * */

    private fun train(epochs: Int): List<Float> {
        var losses = floatArrayOf().toMutableList()

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

            for (epoch in 0 until epochs) {
                val output = model.trainOn(batch)
                val loss = output.get("total_loss") as FloatArray
                losses.add(loss[0])
            }

        } catch (e: ModelBundle.ModelBundleException) {
            print(e)
        } catch (e: Model.ModelException) {
            print(e)
        } catch (e: IOException) {
            print(e)
        }

        return losses
    }
}