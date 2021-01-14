package ai.doc.tensorio.webinardemo

import ai.doc.tensorio.core.data.Batch
import ai.doc.tensorio.core.model.Model
import ai.doc.tensorio.core.modelbundle.ModelBundle
import ai.doc.tensorio.core.training.TrainableModel
import ai.doc.tensorio.core.utilities.AndroidAssets
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.IOException

class MainActivity : AppCompatActivity() {

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

        try {
            // Prepare Model

            val bundle = bundleForFile("keras-cifar10-mobilenet-estimator-train.tiobundle")
            val model = bundle!!.newModel() as TrainableModel

            // Prepare Images and Labels

            val stream1 = applicationContext.assets.open("images/cifar-car.jpg")
            val bitmap1 = BitmapFactory.decodeStream(stream1)
            val label1 = intArrayOf(0)

            val stream2 = applicationContext.assets.open("images/cifar-bird.jpg")
            val bitmap2 = BitmapFactory.decodeStream(stream2)
            val label2 = intArrayOf(2)

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
    }
}