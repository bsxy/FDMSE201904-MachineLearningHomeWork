/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.posenet

import android.Manifest
import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.support.v4.app.ActivityCompat
import android.support.v4.app.DialogFragment
import android.support.v4.app.Fragment
import android.support.v4.content.ContextCompat
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.view.LayoutInflater
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.pow
import org.tensorflow.lite.examples.posenet.lib.BodyPart
import org.tensorflow.lite.examples.posenet.lib.Person
import org.tensorflow.lite.examples.posenet.lib.Posenet
import org.tensorflow.lite.examples.posenet.lib.Position
import java.io.*

enum class BodyState {
  UNKNOWN,
  NORMAL,
  FALL
}

class PosenetActivity :
  Fragment(),
  ActivityCompat.OnRequestPermissionsResultCallback {

  /** List of body joints that should be connected.    */
  private val bodyJoints = listOf(
    Pair(BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW),
    Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER),
    Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
    Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
    Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
    Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
    Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
    Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER),
    Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
    Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
    Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
  )

  /** Threshold for confidence score. */
  private val minConfidence = 0.5

  /** Radius of circle used to draw keypoints.  */
  private val circleRadius = 8.0f

  /** Paint class holds the style and color information to draw geometries,text and bitmaps. */
  private var paint = Paint()

  /** A shape for extracting frame data.   */
  private val PREVIEW_WIDTH = 640
  private val PREVIEW_HEIGHT = 480

  /** An object for the Posenet library.    */
  private lateinit var posenet: Posenet

  /** ID of the current [CameraDevice].   */
  private var cameraId: String? = null

  /** A [SurfaceView] for camera preview.   */
  private var surfaceView: SurfaceView? = null

  /** A [CameraCaptureSession] for camera preview.   */
  private var captureSession: CameraCaptureSession? = null

  /** A reference to the opened [CameraDevice].    */
  private var cameraDevice: CameraDevice? = null

  /** The [android.util.Size] of camera preview.  */
  private var previewSize: Size? = null

  /** The [android.util.Size.getWidth] of camera preview. */
  private var previewWidth = 0

  /** The [android.util.Size.getHeight] of camera preview.  */
  private var previewHeight = 0

  /** A counter to keep count of total frames.  */
  private var frameCounter = 0

  /** An IntArray to save image data in ARGB8888 format  */
  private lateinit var rgbBytes: IntArray

  /** A ByteArray to save image data in YUV format  */
  private var yuvBytes = arrayOfNulls<ByteArray>(3)

  /** An additional thread for running tasks that shouldn't block the UI.   */
  private var backgroundThread: HandlerThread? = null

  /** A [Handler] for running tasks in the background.    */
  private var backgroundHandler: Handler? = null

  /** An [ImageReader] that handles preview frame capture.   */
  private var imageReader: ImageReader? = null

  /** [CaptureRequest.Builder] for the camera preview   */
  private var previewRequestBuilder: CaptureRequest.Builder? = null

  /** [CaptureRequest] generated by [.previewRequestBuilder   */
  private var previewRequest: CaptureRequest? = null

  /** A [Semaphore] to prevent the app from exiting before closing the camera.    */
  private val cameraOpenCloseLock = Semaphore(1)

  /** Whether the current camera device supports Flash or not.    */
  private var flashSupported = false

  /** Orientation of the camera sensor.   */
  private var sensorOrientation: Int? = null

  /** Abstract interface to someone holding a display surface.    */
  private var surfaceHolder: SurfaceHolder? = null

  /** [CameraDevice.StateCallback] is called when [CameraDevice] changes its state.   */
  private val stateCallback = object : CameraDevice.StateCallback() {

    override fun onOpened(cameraDevice: CameraDevice) {
      cameraOpenCloseLock.release()
      this@PosenetActivity.cameraDevice = cameraDevice
      createCameraPreviewSession()
    }

    override fun onDisconnected(cameraDevice: CameraDevice) {
      cameraOpenCloseLock.release()
      cameraDevice.close()
      this@PosenetActivity.cameraDevice = null
    }

    override fun onError(cameraDevice: CameraDevice, error: Int) {
      onDisconnected(cameraDevice)
      this@PosenetActivity.activity?.finish()
    }
  }

  /**
   * A [CameraCaptureSession.CaptureCallback] that handles events related to JPEG capture.
   */
  private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
    override fun onCaptureProgressed(
      session: CameraCaptureSession,
      request: CaptureRequest,
      partialResult: CaptureResult
    ) {
    }

    override fun onCaptureCompleted(
      session: CameraCaptureSession,
      request: CaptureRequest,
      result: TotalCaptureResult
    ) {
    }
  }

  /**
   * Shows a [Toast] on the UI thread.
   *
   * @param text The message to show
   */
  private fun showToast(text: String) {
    val activity = activity
    activity?.runOnUiThread { Toast.makeText(activity, text, Toast.LENGTH_SHORT).show() }
  }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? = inflater.inflate(R.layout.activity_posenet, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    surfaceView = view.findViewById(R.id.surfaceView)
    surfaceHolder = surfaceView!!.holder
  }

  override fun onResume() {
    super.onResume()
    startBackgroundThread()
  }

  override fun onStart() {
    super.onStart()
    openCamera()
    posenet = Posenet(this.context!!)
    m_celues = readCelue();
  }

  override fun onPause() {
    closeCamera()
    stopBackgroundThread()
    super.onPause()
  }

  override fun onDestroy() {
    super.onDestroy()
    posenet.close()
  }

  private fun requestCameraPermission() {
    if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
      ConfirmationDialog().show(childFragmentManager, FRAGMENT_DIALOG)
    } else {
      requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
    }
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<String>,
    grantResults: IntArray
  ) {
    if (requestCode == REQUEST_CAMERA_PERMISSION) {
      if (grantResults.size != 1 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
        ErrorDialog.newInstance(getString(R.string.request_permission))
          .show(childFragmentManager, FRAGMENT_DIALOG)
      }
    } else {
      super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }
  }

  /**
   * Sets up member variables related to camera.
   */
  private fun setUpCameraOutputs() {

    val activity = activity
    val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    try {
      for (cameraId in manager.cameraIdList) {
        val characteristics = manager.getCameraCharacteristics(cameraId)

        // We don't use a front facing camera in this sample.
        val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
        if (cameraDirection != null &&
          cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
        ) {
          continue
        }

        previewSize = Size(PREVIEW_WIDTH, PREVIEW_HEIGHT)

        imageReader = ImageReader.newInstance(
          PREVIEW_WIDTH, PREVIEW_HEIGHT,
          ImageFormat.YUV_420_888, /*maxImages*/ 2
        )

        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)!!

        previewHeight = previewSize!!.height
        previewWidth = previewSize!!.width

        // Initialize the storage bitmaps once when the resolution is known.
        rgbBytes = IntArray(previewWidth * previewHeight)

        // Check if the flash is supported.
        flashSupported =
          characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true

        this.cameraId = cameraId

        // We've found a viable camera and finished setting up member variables,
        // so we don't need to iterate through other available cameras.
        return
      }
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    } catch (e: NullPointerException) {
      // Currently an NPE is thrown when the Camera2API is used but not supported on the
      // device this code runs.
      ErrorDialog.newInstance(getString(R.string.camera_error))
        .show(childFragmentManager, FRAGMENT_DIALOG)
    }
  }

  /**
   * Opens the camera specified by [PosenetActivity.cameraId].
   */
  private fun openCamera() {
    val permissionCamera = ContextCompat.checkSelfPermission(activity!!, Manifest.permission.CAMERA)
    if (permissionCamera != PackageManager.PERMISSION_GRANTED) {
      requestCameraPermission()
    }
    setUpCameraOutputs()
    val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    try {
      // Wait for camera to open - 2.5 seconds is sufficient
      if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
        throw RuntimeException("Time out waiting to lock camera opening.")
      }
      manager.openCamera(cameraId!!, stateCallback, backgroundHandler)
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    } catch (e: InterruptedException) {
      throw RuntimeException("Interrupted while trying to lock camera opening.", e)
    }
  }

  /**
   * Closes the current [CameraDevice].
   */
  private fun closeCamera() {
    if (captureSession == null) {
      return
    }

    try {
      cameraOpenCloseLock.acquire()
      captureSession!!.close()
      captureSession = null
      cameraDevice!!.close()
      cameraDevice = null
      imageReader!!.close()
      imageReader = null
    } catch (e: InterruptedException) {
      throw RuntimeException("Interrupted while trying to lock camera closing.", e)
    } finally {
      cameraOpenCloseLock.release()
    }
  }

  /**
   * Starts a background thread and its [Handler].
   */
  private fun startBackgroundThread() {
    backgroundThread = HandlerThread("imageAvailableListener").also { it.start() }
    backgroundHandler = Handler(backgroundThread!!.looper)
  }

  /**
   * Stops the background thread and its [Handler].
   */
  private fun stopBackgroundThread() {
    backgroundThread?.quitSafely()
    try {
      backgroundThread?.join()
      backgroundThread = null
      backgroundHandler = null
    } catch (e: InterruptedException) {
      Log.e(TAG, e.toString())
    }
  }

  /** Fill the yuvBytes with data from image planes.   */
  private fun fillBytes(planes: Array<Image.Plane>, yuvBytes: Array<ByteArray?>) {
    // Row stride is the total number of bytes occupied in memory by a row of an image.
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (i in planes.indices) {
      val buffer = planes[i].buffer
      if (yuvBytes[i] == null) {
        yuvBytes[i] = ByteArray(buffer.capacity())
      }
      buffer.get(yuvBytes[i]!!)
    }
  }

  /** A [OnImageAvailableListener] to receive frames as they are available.  */
  private var imageAvailableListener = object : OnImageAvailableListener {
    override fun onImageAvailable(imageReader: ImageReader) {
      // We need wait until we have some size from onPreviewSizeChosen
      if (previewWidth == 0 || previewHeight == 0) {
        return
      }

      val image = imageReader.acquireLatestImage() ?: return
      fillBytes(image.planes, yuvBytes)

      ImageUtils.convertYUV420ToARGB8888(
        yuvBytes[0]!!,
        yuvBytes[1]!!,
        yuvBytes[2]!!,
        previewWidth,
        previewHeight,
        /*yRowStride=*/ image.planes[0].rowStride,
        /*uvRowStride=*/ image.planes[1].rowStride,
        /*uvPixelStride=*/ image.planes[1].pixelStride,
        rgbBytes
      )

      // Create bitmap from int array
      val imageBitmap = Bitmap.createBitmap(
        rgbBytes, previewWidth, previewHeight,
        Bitmap.Config.ARGB_8888
      )

      // Create rotated version for portrait display
      val rotateMatrix = Matrix()
      rotateMatrix.postRotate(90.0f)

      val rotatedBitmap = Bitmap.createBitmap(
        imageBitmap, 0, 0, previewWidth, previewHeight,
        rotateMatrix, true
      )
      image.close()

      // Process an image for analysis in every 3 frames.
      frameCounter = (frameCounter + 1) % 3
      if (frameCounter == 0) {
        processImage(rotatedBitmap)
      }
    }
  }

  /** Crop Bitmap to maintain aspect ratio of model input.   */
  private fun cropBitmap(bitmap: Bitmap): Bitmap {
    val bitmapRatio = bitmap.height.toFloat() / bitmap.width
    val modelInputRatio = MODEL_HEIGHT.toFloat() / MODEL_WIDTH
    var croppedBitmap = bitmap

    // Acceptable difference between the modelInputRatio and bitmapRatio to skip cropping.
    val maxDifference = 1e-5

    // Checks if the bitmap has similar aspect ratio as the required model input.
    when {
      abs(modelInputRatio - bitmapRatio) < maxDifference -> return croppedBitmap
      modelInputRatio < bitmapRatio -> {
        // New image is taller so we are height constrained.
        val cropHeight = bitmap.height - (bitmap.width.toFloat() / modelInputRatio)
        croppedBitmap = Bitmap.createBitmap(
          bitmap,
          0,
          (cropHeight / 2).toInt(),
          bitmap.width,
          (bitmap.height - cropHeight).toInt()
        )
      }
      else -> {
        val cropWidth = bitmap.width - (bitmap.height.toFloat() * modelInputRatio)
        croppedBitmap = Bitmap.createBitmap(
          bitmap,
          (cropWidth / 2).toInt(),
          0,
          (bitmap.width - cropWidth).toInt(),
          bitmap.height
        )
      }
    }
    return croppedBitmap
  }

  /** Set the paint color and size.    */
  private fun setPaint() {
    paint.color = Color.RED
    paint.textSize = 80.0f
    paint.strokeWidth = 8.0f
  }


  private fun getBodyPosByPart(part: BodyPart, person: Person): org.tensorflow.lite.examples.posenet.lib.Position? {
    for (keyPoint in person.keyPoints) {
      if (keyPoint.bodyPart == part && keyPoint.score > minConfidence) {
        return keyPoint.position
      }
    }
    return null
  }

  var state_name: Array<String> = arrayOf(
     "未检测出人体",
    "正常姿态",
    "异常姿态"
  )
  var body_name: Array<String> = arrayOf(
    "NOSE", "鼻子",
    "LEFT_EYE", "左眼",
    "RIGHT_EYE", "右眼",
    "LEFT_EAR", "左耳",
    "RIGHT_EAR", "右耳",
    "LEFT_SHOULDER", "左肩",
    "RIGHT_SHOULDER", "右肩",
    "LEFT_ELBOW", "左肘",
    "RIGHT_ELBOW", "右肘",
    "LEFT_WRIST", "左腕",
    "RIGHT_WRIST", "右腕",
    "LEFT_HIP", "左髋",
    "RIGHT_HIP", "右髋",
    "LEFT_KNEE", "左膝",
    "RIGHT_KNEE", "右膝",
    "LEFT_ANKLE", "左踝",
    "RIGHT_ANKLE", "右踝"
  )
  var m_celues :ArrayList<IntArray>? = null

  /** Draw bitmap on Canvas.   */
  private fun draw(canvas: Canvas, person: Person, bitmap: Bitmap) {
    val screenWidth: Int = canvas.width
    val screenHeight: Int = canvas.height
    setPaint()
    canvas.drawBitmap(
      bitmap,
      Rect(0, 0, previewHeight, previewWidth),
      Rect(0, 0, screenWidth, screenHeight),
      paint
    )

    val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
    val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT

    // Draw key points over the image.
    for (keyPoint in person.keyPoints) {
      if (keyPoint.score > minConfidence) {
        paint.color = Color.RED
        paint.textSize = 80.0f
        paint.strokeWidth = 8.0f
        val position = keyPoint.position
        val adjustedX: Float = position.x.toFloat() * widthRatio
        val adjustedY: Float = position.y.toFloat() * heightRatio
        canvas.drawCircle(adjustedX, adjustedY, circleRadius, paint)
        paint.color = Color.RED
        paint.textSize = 30.0f
        paint.strokeWidth = 2.0f
        canvas.drawText(
          "%s".format(body_name[keyPoint.bodyPart.ordinal * 2 + 1]),
          //adjustedX.toInt(), adjustedY.toInt()),
          adjustedX.toFloat() + 5,
          adjustedY.toFloat() + 15,
          paint
        )
      }

      val position = keyPoint.position

      if (keyPoint.score > minConfidence) {
        paint.color = Color.RED
      } else {
        paint.color = Color.GRAY
      }

      paint.textSize = 30.0f
      paint.strokeWidth = 2.0f
      canvas.drawText(
        "%s(%.2f) %03d,%03d".format(
          body_name[keyPoint.bodyPart.ordinal * 2 + 1], keyPoint.score,
          position.x.toInt(), position.y.toInt()
        ),
        10.0f + (screenWidth / 2) * (keyPoint.bodyPart.ordinal % 2),
        ((screenHeight.toFloat() - 10 - 20.0f * ((16 - keyPoint.bodyPart.ordinal)))),
        paint
      )
    }
    paint.color = Color.RED
    paint.textSize = 80.0f
    paint.strokeWidth = 8.0f
    for (line in bodyJoints) {
      if (
        (person.keyPoints[line.first.ordinal].score > minConfidence) and
        (person.keyPoints[line.second.ordinal].score > minConfidence)
      ) {
        canvas.drawLine(
          person.keyPoints[line.first.ordinal].position.x.toFloat() * widthRatio,
          person.keyPoints[line.first.ordinal].position.y.toFloat() * heightRatio,
          person.keyPoints[line.second.ordinal].position.x.toFloat() * widthRatio,
          person.keyPoints[line.second.ordinal].position.y.toFloat() * heightRatio,
          paint
        )
      }
    }

    var body_state = this.judgeState(person,m_celues!!)


    canvas.drawText(
      "人体状态: "+ state_name[body_state.ordinal],
      (15.0f * widthRatio),
      (190.0f * heightRatio),
      paint
    )


    paint.color = Color.RED
    paint.textSize = 30.0f
    paint.strokeWidth = 2.0f
    canvas.drawText(
      "Ratio: %.2f，%.2f Model(w/h):%d,%d Canvas(w/h):%.2f,%.2f".format(
        widthRatio, heightRatio,
        MODEL_WIDTH, MODEL_HEIGHT,
        canvas.width.toFloat(), canvas.height.toFloat()
      ),
      (15.0f * widthRatio),
      (203.0f * heightRatio),
      paint
    )
    paint.color = Color.RED
    paint.textSize = 80.0f
    paint.strokeWidth = 8.0f
//    canvas.drawText(
//      "Score: %.2f".format(person.score),
//      (15.0f * widthRatio),
//      (233.0f * heightRatio),
//      paint
//    )
//    canvas.drawText(
//      "Device: %s".format(posenet.device),
//      (15.0f * widthRatio),
//      (243.0f * heightRatio),
//      paint
//    )
//    canvas.drawText(
//      "Time: %.2f ms".format(posenet.lastInferenceTimeNanos * 1.0f / 1_000_000),
//      (15.0f * widthRatio),
//      (253.0f * heightRatio),
//      paint
//    )

    // Draw!
    surfaceHolder!!.unlockCanvasAndPost(canvas)
  }

  private fun calcBearing(s:Position,e:Position):Int{
    val s_y = MODEL_HEIGHT-s.y
    val e_y = MODEL_HEIGHT-e.y
    if(s.x == e.x){
      if(s_y>e_y){
        return 270
      }else{
        return 90
      }
    }

    if(s_y ==e_y){
      if(s.x >e.x){
        return 180
      }else{
        return 0
      }
    }

    var dx = (e.x-s.x).toDouble()
    var dy = (e_y-s_y).toDouble()
    var angle = Math.atan(Math.abs(dy/dx))
    if(dx > 0 && dy > 0){
      angle = angle
    }else if(dx >0 && dy < 0){
      angle = Math.PI*2-angle
    }else if(dx < 0 && dy >0){
      angle = Math.PI-angle
    }else {
      angle = Math.PI+angle
    }

    return (angle*180/Math.PI).toInt()

  }
  private fun readCelue(): ArrayList<IntArray>{
    val celues = ArrayList<IntArray>()

    try {
      var line: String?
      //val fileDescriptor = this.context!!.assets.openFd("celue-data.csv")
      //val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
      var assetManager = context!!.getAssets()
      var inputStream: InputStream? = null
      inputStream = assetManager.open("celue-data.csv")
      //fileReader = BufferedReader(FileReader("celue-data.csv"))
      var fileReader = BufferedReader(InputStreamReader(inputStream, "UTF-8"))

      // Read CSV header
      fileReader.readLine()

      // Read the file line by line starting from the second line
      line = fileReader.readLine()
      line = fileReader.readLine()
      while (line != null) {
        Log.i("mytest", line)
        val tokens = line.split(",")
        assert(tokens.size>=19)
        var item = IntArray(17)
        Log.i("mytest",item.toString())
        Log.i("mytest",item.size.toString())
        Log.i("mytest",tokens.size.toString())

        for (i in item.indices){
          if(tokens[i+2].length >0) {
            item[i] = Integer.parseInt(tokens[i+2])
          }else{
            item[i] = 0
          }
          Log.i("mytest", i.toString() + "-"+item[i].toString())
        }
        celues.add(item)

        line = fileReader.readLine()

      }

      // Print the new customer list
      for (customer in celues) {
        println(customer)
      }
    } catch (e: Exception) {
      println("Reading CSV Error!")
      e.printStackTrace()
    } finally {
    }

    return celues
  }
  private fun judgeState(person:Person,rule_mt:ArrayList<IntArray>) : BodyState{
    if(person.score  < minConfidence)return BodyState.UNKNOWN;

    for (row_ind in rule_mt.indices){
      var row_item = rule_mt[row_ind]
      for(col_ind in row_item.indices){
        var item = row_item[col_ind];
        if(item >0){
          var pt_s = getBodyPosByPart(BodyPart.values()[row_ind],person)
          var pt_e = getBodyPosByPart(BodyPart.values()[col_ind],person)
          if(pt_s !=null && pt_e!=null){
            var brg_angle = calcBearing(pt_s,pt_e);
            if(abs(brg_angle-item)> 40){
              Log.i("mytest",brg_angle.toString() + "-" + item.toString() +" " + body_name[row_ind*2+1]+"->"
                      +body_name[col_ind*2+1])
              return BodyState.FALL
            }
          }
        }
      }
    }
    return BodyState.NORMAL

    //1st version
//    var l_wrist = getBodyPosByPart(BodyPart.LEFT_WRIST,person)
//    var r_wrist = getBodyPosByPart(BodyPart.RIGHT_WRIST,person)
//
//    var l_ankle_pt = getBodyPosByPart(BodyPart.LEFT_ANKLE,person)
//    var r_ankel_pt = getBodyPosByPart(BodyPart.RIGHT_ANKLE,person)
//
//    var wrist_pt:org.tensorflow.lite.examples.posenet.lib.Position? = null
//      if(l_wrist_pt!=null && r_wrist_pt!=null){
//        if(l_wrist_pt.x < r_wrist_pt.x)
//          wrist_pt = l_wrist_pt
//        else
//          wrist_pt = r_wrist_pt
//      }
//      else if(l_wrist_pt!=null){
//        wrist_pt = l_wrist_pt
//      }
//      else if(r_wrist_pt!=null){
//        wrist_pt = r_wrist_pt
//      }
//
//
//    var ankel_pt:org.tensorflow.lite.examples.posenet.lib.Position? = null
//      if(l_ankle_pt!=null && r_ankel_pt!=null){
//        if(l_ankle_pt.x < r_ankel_pt.x)
//          ankel_pt = l_ankle_pt
//        else
//          ankel_pt = r_ankel_pt
//      }
//      else if(l_ankle_pt!=null){
//        ankel_pt = l_ankle_pt
//      }
//      else if(r_ankel_pt!=null){
//        ankel_pt = r_ankel_pt
//      }
//
//
//    if(wrist_pt!=null && ankel_pt!=null ){
//      body_state = BodyState.FALL
//      }
  }

  /** Process image using Posenet library.   */
  private fun processImage(bitmap: Bitmap) {
    // Crop bitmap.
    val croppedBitmap = cropBitmap(bitmap)

    // Created scaled version of bitmap for model input.
    val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, MODEL_WIDTH, MODEL_HEIGHT, true)

    // Perform inference.
    val person = posenet.estimateSinglePose(scaledBitmap)
    val canvas: Canvas = surfaceHolder!!.lockCanvas()
    draw(canvas, person, bitmap)
  }

  /**
   * Creates a new [CameraCaptureSession] for camera preview.
   */
  private fun createCameraPreviewSession() {
    try {

      // We capture images from preview in YUV format.
      imageReader = ImageReader.newInstance(
        previewSize!!.width, previewSize!!.height, ImageFormat.YUV_420_888, 2
      )
      imageReader!!.setOnImageAvailableListener(imageAvailableListener, backgroundHandler)

      // This is the surface we need to record images for processing.
      val recordingSurface = imageReader!!.surface

      // We set up a CaptureRequest.Builder with the output Surface.
      previewRequestBuilder = cameraDevice!!.createCaptureRequest(
        CameraDevice.TEMPLATE_PREVIEW
      )
      previewRequestBuilder!!.addTarget(recordingSurface)

      // Here, we create a CameraCaptureSession for camera preview.
      cameraDevice!!.createCaptureSession(
        listOf(recordingSurface),
        object : CameraCaptureSession.StateCallback() {
          override fun onConfigured(cameraCaptureSession: CameraCaptureSession) {
            // The camera is already closed
            if (cameraDevice == null) return

            // When the session is ready, we start displaying the preview.
            captureSession = cameraCaptureSession
            try {
              // Auto focus should be continuous for camera preview.
              previewRequestBuilder!!.set(
                CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
              )
              // Flash is automatically enabled when necessary.
              setAutoFlash(previewRequestBuilder!!)

              // Finally, we start displaying the camera preview.
              previewRequest = previewRequestBuilder!!.build()
              captureSession!!.setRepeatingRequest(
                previewRequest!!,
                captureCallback, backgroundHandler
              )
            } catch (e: CameraAccessException) {
              Log.e(TAG, e.toString())
            }
          }

          override fun onConfigureFailed(cameraCaptureSession: CameraCaptureSession) {
            showToast("Failed")
          }
        },
        null
      )
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    }
  }

  private fun setAutoFlash(requestBuilder: CaptureRequest.Builder) {
    if (flashSupported) {
      requestBuilder.set(
        CaptureRequest.CONTROL_AE_MODE,
        CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH
      )
    }
  }

  /**
   * Shows an error message dialog.
   */
  class ErrorDialog : DialogFragment() {

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
      AlertDialog.Builder(activity)
        .setMessage(arguments!!.getString(ARG_MESSAGE))
        .setPositiveButton(android.R.string.ok) { _, _ -> activity!!.finish() }
        .create()

    companion object {

      @JvmStatic
      private val ARG_MESSAGE = "message"

      @JvmStatic
      fun newInstance(message: String): ErrorDialog = ErrorDialog().apply {
        arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
      }
    }
  }

  companion object {
    /**
     * Conversion from screen rotation to JPEG orientation.
     */
    private val ORIENTATIONS = SparseIntArray()
    private val FRAGMENT_DIALOG = "dialog"

    init {
      ORIENTATIONS.append(Surface.ROTATION_0, 90)
      ORIENTATIONS.append(Surface.ROTATION_90, 0)
      ORIENTATIONS.append(Surface.ROTATION_180, 270)
      ORIENTATIONS.append(Surface.ROTATION_270, 180)
    }

    /**
     * Tag for the [Log].
     */
    private const val TAG = "PosenetActivity"
  }
}
