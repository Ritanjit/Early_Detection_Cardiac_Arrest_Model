# 💓 Early Detection of Cardiac Arrest using Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uRpSNWPYPE__Nhhtnl6-anwPlyJU5OGz#scrollTo=vWpy8lJWN45l)
[![Python](https://img.shields.io/badge/Python_3.10_+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-3.50+-orange?logo=gradio)](https://www.gradio.app)
[![Hugging Face Spaces](https://img.shields.io/badge/Deploy-HF_Spaces-blue?logo=huggingface)](https://huggingface.co/spaces)

---

> **Project Description:**
>
> This project implements an end-to-end pipeline for **early detection of cardiac arrest** using ECG signal classification. The system utilizes a deep learning architecture (CNN) and provides a lightweight quantized model for efficient deployment on resource-constrained edge devices.

## 🚀 Live Demo

Experience the model live in a browser using [Gradio Interface](https://c908c6d5e34fb30145.gradio.live/). You can paste or upload a 360-sample ECG signal and get instant heartbeat classification results:

**Classes:**

* `N`: Normal beat
* `L`: Left bundle branch block beat
* `R`: Right bundle branch block beat
* `A`: Atrial premature beat
* `V`: Ventricular ectopic beat

## 📊 Dataset

* **Source:** [MIT-BIH Arrhythmia Dataset](https://physionet.org/content/mitdb/1.0.0/)
* **Preprocessing:**

  * Denoising with Discrete Wavelet Transform (DWT)
  * Z-score normalization
  * Window slicing around R-peak (360 samples)

## 🧠 Model Architectures

We built and compared the following models:

### ✅ CNN (Baseline)

* 5 Conv1D + AvgPooling layers
* Dense + Dropout
* Output: Softmax over 5 classes

### ✅ Pruned CNN

* 50% sparsity pruning with TensorFlow Model Optimization Toolkit
* Same accuracy, reduced size & latency

### ✅ Quantized CNN

* Post-training INT8 quantization
* 96% smaller, 10x faster inference

---

## 📈 Results

<table>
<tr>
<td align="center">

#### MODEL BENCHMARKS

</td>
</tr>

<tr>
<td>

  <img src="https://raw.githubusercontent.com/Ritanjit/Lightweight_Model_Early_Detection_Cardiac_Arrest/main/model_comparision.png" width="1100"/>

   _These plots provide a clear view of trade-offs between performance and efficiency for deployment._

</td>
</tr>
</table>


| Model         | Accuracy | Latency (ms) | Size (MB) |
| ------------- | -------- | ------------ | --------- |
| Original CNN  | 0.974    | 11.24        | 5.9       |
| Pruned CNN    | 0.971    | 8.87         | 3.0       |
| Quantized CNN | 0.970    | 1.07         | 0.34      |

---

## 🛠 How to Use

### 🔧 Setup

```bash
conda create -n cardiac python=3.10
conda activate cardiac
pip install -r requirements.txt
```

### 💻 Run Locally

```bash
python app.py     # Launch Gradio interface locally
```

### ☁️ Deploy Permanently

```bash
gradio deploy     # Uploads to Hugging Face Spaces
```

---

## 📦 Deployment Ready

The quantized `.tflite` model can be deployed to:

* Mobile (via TensorFlow Lite)
* Microcontrollers (with Edge Impulse / TinyML)
* Raspberry Pi / NVIDIA Jetson

---

## 🧪 How to Test

Simply paste or upload a **360-point ECG beat** into the Gradio app. The model will preprocess, normalize, and classify the beat in real-time.

---

## 🔭 Future Work

* ✅ Add support for more heartbeat types
* ✅ Convert to TFLite with Edge TPU compatibility
* 🔄 Real-time ECG streaming integration
* 🌐 API deployment via FastAPI / Streamlit

---

## 🙏 Acknowledgements

* [MIT-BIH Dataset - PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
* [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
* [Gradio](https://gradio.app/)

---

<div align="center">

Made with ❤️ during a summer research internship at [IIIT Guwahati](https://www.iiitg.ac.in/) by [Ritanjit Das](https://github.com/ritanjit)

</div>
