# Shadowcr3 Synapse

PROJECT SPECIFICATION · VERSION 1.0 · NOVEMBER 2026

[Neural Network Nodes Deep Learning Artificial Intelligence Machine Learning Model](https://images.openai.com/static-rsc-4/-f5VmUOzmYlqnaAiW-qz6Q1wmDeGIAO11Xx7mmurAq5LlKW9S0vzmJgsRNmpKGDwHLupdhvY2EMI4oHXbNa5iJ2e4cyiLv6_Sk9UWRvVt4Wss6z_xad7Z0wg-k9R-sogh9G6Eo-E5HnsmpcyRPkj3fszUCeos3FZdD9_G7GNV6LAb0UdJmY352fZ0Sn8nq7-?purpose=inline)

Project identity

Approved concept

Name: Shadowcr3 Synapse

Category: Artificial Intelligence / Machine Learning

Project type: Custom neural network

Primary objective: Handwritten digit recognition

Recognition range: 1–9

Implementation: From scratch

Development language: Python

Philosophy: Understand, implement, train, evaluate and deploy.

## 1. Project vision

Shadowcr3 Synapse is a lightweight handwritten digit recognition system designed to demonstrate the fundamental principles of artificial neural networks through an independently implemented learning engine.

The system will allow users to draw a handwritten number on a digital canvas and receive a prediction from a trained neural network.

The project is not intended to compete with large AI models. Its purpose is to demonstrate that a relatively small neural network can learn meaningful patterns from data through mathematical optimization.

### Final vision

A fully functional desktop/web-accessible application where:

1. A user draws a digit.
2. The application captures the drawing.
3. The image is processed into numerical input.
4. The neural network evaluates the input.
5. The system predicts the digit.
6. The application displays the prediction and confidence.
7. The system can explain its prediction through a visualization of output probabilities.

# 2. Final goals and objectives

| ID  | Goal                                | Success criteria                              |
| --- | ----------------------------------- | --------------------------------------------- |
| G01 | Build a neural network from scratch | No prebuilt neural-network training framework |
| G02 | Implement forward propagation       | Correct mathematical outputs                  |
| G03 | Implement backpropagation           | Verified gradients                            |
| G04 | Train on handwritten digits         | Training completes successfully               |
| G05 | Recognize digits 1–9                | Correct classification                        |
| G06 | Build drawing interface             | Interactive canvas                            |
| G07 | Integrate model and interface       | Real-time prediction                          |
| G08 | Evaluate reliability                | Independent test dataset                      |
| G09 | Document mathematics                | Complete technical documentation              |
| G10 | Make it reproducible                | Repeatable training and evaluation            |

### Performance targets

These are engineering targets, not guaranteed results.

- Test accuracy: target 97% or higher on the filtered MNIST test set.
- Prediction latency: target under 100 ms on a normal laptop CPU.
- Model size: target under 5 MB.
- Application: functional without an internet connection after setup.
- Training: reproducible using saved configuration and random seeds.

The accuracy target applies to the test dataset, not every possible handwritten digit.

# 3. Product requirements document (PRD)

PRD · Product specification

## 3.1 Problem statement

Most people interact with AI through existing pretrained models without understanding the mechanisms responsible for learning.

DigitNet addresses this by creating a small, understandable AI system that learns handwritten digit patterns using an independently implemented neural network.

## 3.2 Target users

- Project developer and researcher.
- Students learning machine learning.
- Users experimenting with handwritten digit recognition.
- Future contributors to Shadowcr3 AI research.

## 3.3 Functional requirements

| ID   | Requirement                        | Priority |
| ---- | ---------------------------------- | -------- |
| FR01 | Accept handwritten input           | Critical |
| FR02 | Clear and reset canvas             | Critical |
| FR03 | Convert drawing into 28×28 image   | Critical |
| FR04 | Normalize image pixels             | Critical |
| FR05 | Predict digits 1–9                 | Critical |
| FR06 | Display prediction                 | Critical |
| FR07 | Display all nine output scores     | High     |
| FR08 | Save trained model                 | Critical |
| FR09 | Load model without retraining      | Critical |
| FR10 | Display prediction processing time | Medium   |
| FR11 | Show training metrics              | High     |
| FR12 | Support offline operation          | High     |

## 3.4 Non-functional requirements

- Lightweight and CPU-friendly.
- Modular architecture.
- Reproducible training.
- Readable and documented code.
- No unnecessary dependencies.
- No external API dependency for prediction.
- Clear separation between training and inference.
- Robust handling of empty and invalid input.

## 3.5 User workflow

[New MNIST Digit Classifier Web App on Hugging Face | Abdul Moiz Meer posted on the topic | LinkedIn](https://images.openai.com/static-rsc-4/cx_RSPm_o1yjrArzdeTf-0NGfqVIzIW0IgIzYTkAVYp0HSQh7S4Rxn7pPAX1LRE_SI4FI3D75dkNwYGc1n1So2mgE4ZeTJ7b4MoUsGoV_2df0YAnGxYzbokz3SLcgW2fOToYoOjPqdBPpB3gKxyc6CqTqww6mXleTz59Fwcj-XM?purpose=inline)

[linkedin.com](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A7396094546650075137/?utm_source=chatgpt.com)

1. User opens DigitNet.
2. User draws a digit.
3. User presses Predict.
4. System processes the image.
5. Model returns nine output scores.
6. Interface displays the predicted digit.
7. User can clear the canvas and try again.

## 3.6 Out-of-scope features

Version 1.0 will not include:

- Handwriting sentence recognition.
- Alphabet recognition.
- Voice recognition.
- Large language models.
- Cloud-based inference.
- Mobile applications.
- Multi-user accounts.
- Automatic retraining from user drawings.

These can be considered for future versions.

# 4. Technical requirements document (TRD)

TRD · Engineering specification

## 4.1 Technology stack

| Layer                 | Technology              |
| --------------------- | ----------------------- |
| Core language         | Python 3                |
| Numerical computation | NumPy                   |
| Dataset               | MNIST                   |
| Training engine       | Custom implementation   |
| Visualization         | Matplotlib              |
| Backend               | Flask                   |
| Frontend              | HTML5, CSS3, JavaScript |
| Model storage         | NumPy NPZ               |
| Configuration         | JSON                    |
| Testing               | Python unittest         |

Important: NumPy is permitted for matrix calculations. PyTorch, TensorFlow, Keras and other prebuilt neural-network training engines are prohibited in the core implementation.

## 4.2 Neural network architecture

## Synapse V1 architecture

Fully connected feedforward neural network · Multiclass classification

# 784

Input

# 128

Hidden 1

# 64

Hidden 2

# 9

Output

Layer configuration

- Input: 784 numerical pixel values.
- Hidden layer 1: 128 neurons, ReLU.
- Hidden layer 2: 64 neurons, ReLU.
- Output: 9 neurons, Softmax.

  Trainable parameters: 109,385

The parameter count includes weights and biases:

\\[ (784 \times 128 +128) \\]

\\[ +(128 \times 64+64) \\]

\\[ +(64 \times 9+9)=109,385 \\]

This is a deliberately compact architecture.

## 4.3 Mathematical implementation

The developer must implement the following independently.

A. Weighted sum

\\[ z=Wx+b \\]

B. ReLU activation

\\[ ReLU(x)=\max(0,x) \\]

C. Softmax

\\[ P_i=\frac{e^{z_i}}{\sum_j e^{z_j}} \\]

Use numerically stable softmax by subtracting the maximum logit before exponentiation.

D. Cross-entropy loss

\\[ L=-\sum\_{i=1}^{9}y_i\log(P_i) \\]

E. Backpropagation

Implement the chain rule to calculate gradients for every trainable weight and bias.

F. Optimization

Implement Adam or mini-batch gradient descent independently. Adam is the recommended default.

G. Initialization

Use He initialization for the ReLU hidden layers and a suitable small random initialization for the output layer.

## 4.4 Data pipeline

[MNIST Handwritten Digits \[IMAGE\] | EurekAlert! Science News Releases](https://images.openai.com/static-rsc-4/-tOwHF7gmwe1UBINTjbJAcPRPjJpDcaNfolg1zR9U9kX8lXABGHnklFaRslSQXXlRk7t7yVU9pXCWDPUEWgHIPv_ZQ5RqutbNwhFi1BrAxI-N-4-4IrGha9SbxYjX0fPGrFkiUnmqtPZMpak9WhMbzd2UDjLeB6pw34pYLV_JD4?purpose=inline)

[eurekalert.org](https://www.eurekalert.org/multimedia/724624?utm_source=chatgpt.com)

The MNIST dataset contains handwritten digits from 0 to 9. DigitNet will filter out zero.

Pipeline:

1. Load the MNIST dataset.
2. Retain labels 1–9.
3. Separate training and test datasets.
4. Reserve a validation subset from training data.
5. Flatten each 28×28 image into 784 values.
6. Normalize pixel intensities to 0–1.
7. Convert labels into indices 0–8 internally.
8. Train the network.

The original test dataset must remain untouched during training and hyperparameter tuning.

## 4.5 Training configuration

Initial experimental configuration:

| Parameter         | Initial value |
| ----------------- | ------------- |
| Epochs            | 20            |
| Batch size        | 64            |
| Learning rate     | 0.001         |
| Optimizer         | Custom Adam   |
| Hidden activation | ReLU          |
| Output activation | Softmax       |
| Loss              | Cross-entropy |
| Random seed       | 42            |
| Input resolution  | 28×28         |

These are starting values, not immutable settings. We can experiment after establishing a baseline.

## 4.6 Model storage

The trained model should store:

- Weight matrices.
- Bias vectors.
- Architecture metadata.
- Input preprocessing configuration.
- Class mapping.
- Model version.
- Training configuration.

Use `.npz` for numerical parameters and `.json` for metadata.

## 4.7 Application architecture

Web frontend

HTML · CSS · JavaScript · Drawing canvas

HTTP / JSON

Flask backend

Input validation · Image preprocessing · API

Synapse inference engine

Custom NumPy neural network

Prediction response

Predicted class · Nine scores · Processing time

## 4.8 Suggested API

| Endpoint      | Method | Purpose                     |
| ------------- | ------ | --------------------------- |
| `/`           | GET    | Application interface       |
| `/predict`    | POST   | Receive drawing and predict |
| `/health`     | GET    | Check application status    |
| `/model/info` | GET    | Return model metadata       |

Prediction responses should contain the predicted class, output scores and processing time.

# 5. Software design and repository structure

## 5.1 Proposed project structure

```
Shadowcr3-Synapse/
│
├── app/
│   ├── __init__.py
│   ├── server.py
│   ├── routes.py
│   ├── preprocessing.py
│   └── inference.py
│
├── neural_network/
│   ├── __init__.py
│   ├── layers.py
│   ├── activations.py
│   ├── loss.py
│   ├── optimizers.py
│   ├── network.py
│   └── backpropagation.py
│
├── training/
│   ├── dataset.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── metrics.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── models/
│   ├── digitnet_v1.npz
│   └── metadata.json
│
├── experiments/
│   ├── training_logs/
│   └── charts/
│
├── tests/
│   ├── test_activations.py
│   ├── test_gradients.py
│   ├── test_network.py
│   ├── test_preprocessing.py
│   └── test_inference.py
│
├── configs/
│   └── default.json
│
├── docs/
│   ├── PRD.md
│   ├── TRD.md
│   ├── RULES.md
│   └── MATHEMATICS.md
│
├── requirements.txt
├── README.md
├── LICENSE
└── train.py
```

## 5.2 Design language

[Modern analytics dashboard Vectors - Download Free High-Quality Vectors | Magnific (formerly Freepik)](https://images.openai.com/static-rsc-4/XzLc-udnoVFdTEbqANWLw1LR0RVPLOzt3DJe92j8xYrXDEBt0dxYloRansdJohjG2B5EBdRDffkNH6MFeRznJ2QC8j1Kjtxy4XI2NV6qHPo6enwQokrprLvpli2Bu1me9NbvHTZSNKduQP9QKCPzcdgQdGRAuP1VBCWun4oUS5I?purpose=inline)

[Sketch Canvas](https://images.openai.com/static-rsc-4/qAgLX_rd0n1oYrsn1UOw8od8uXrdSQMgP28Vi3xpwvudRW6MEI0RprYmNFN2JRSfOWvyuHRwYJMhofjsQPc63tR_98682U0J1r5-G7ULUfOeVLw7EEe6aehk7UqdquKzaVwPsQECEQrBMKmivBKg44OOBWnStfyapRpmb8yAEFo?purpose=inline)

[Inside ARIA: Teaching a Machine to Think Like a Scientist](https://images.openai.com/static-rsc-4/8L-I2Hq44q5tk7M51wupp5hbPZ2CJNmFV_28EhOz-T9gD4PmxJfl5gp5Y8LfoaO7XvzMdFFmMLgWgMa-3ismNGhJuNB5DFHDyX-7t0FYe5JRecK9GwJ4eyNt1t1M-XfY_Rb0QSpCH85LCQm99xk5zJQRoVmYtGZPQeHFmshtAPA?purpose=inline)

9

The application should have a premium, minimal, technical appearance.

| Element          | Design                 |
| ---------------- | ---------------------- |
| Main background  | Graphite black         |
| Panels           | Dark charcoal          |
| Accent           | Electric blue          |
| Secondary accent | Cyan                   |
| Success          | Green                  |
| Typography       | Inter                  |
| Corners          | Subtle rounded corners |
| Layout           | Clean dashboard        |

### Main application layout

## Synapse

Model loaded

SHADOWCR3 · AI RESEARCH

DRAW YOUR DIGIT

# 7

Drawing canvas

Freehand input

Predict

Clear

PREDICTION

# 7

Example result

Output scores

1

2%

2

3%

3

1%

4

4%

5

3%

6

5%

7

92%

8

2%

9

1%

Illustrative UI only; actual scores come from the trained model.

The actual application will have a functional drawing canvas rather than the illustrative static preview above.

## 5.3 Image preprocessing design

This is particularly important.

A user may draw a large digit, a small digit, or a digit positioned toward one edge of the canvas.

The preprocessing system must:

1. Convert the drawing to grayscale.
2. Invert colors if required to match MNIST.
3. Detect the digit's bounding box.
4. Crop the digit while preserving its proportions.
5. Resize it to fit a 20×20 region.
6. Place it centrally on a 28×28 canvas.
7. Normalize pixel values.
8. Flatten the image into 784 inputs.

We should compare this preprocessing against MNIST samples rather than assuming that resizing alone will work.

## 5.4 Prediction behavior

The system must not pretend every drawing is a valid digit.

For example:

- Empty canvas: reject the input.
- Invalid image: return an input error.
- Ambiguous drawing: display a low-confidence warning.
- Valid drawing: return the most likely digit.

Softmax scores are not automatically calibrated probabilities. We should evaluate calibration before presenting them as reliable confidence percentages.

# 6. Development rules and engineering standards

Mandatory project rules

These rules govern the entire project.

### Rule 1 — No black-box neural network engines

The neural network must be implemented independently.

### Rule 2 — Mathematical transparency

Every major mathematical operation must have an understandable implementation and documentation.

### Rule 3 — No fabricated performance

Never claim an accuracy percentage without testing it on an independent dataset.

### Rule 4 — Reproducibility

Training configurations, random seeds, dataset splits and model versions must be recorded.

### Rule 5 — Modular architecture

The training engine, inference engine, backend and frontend must remain separate.

### Rule 6 — Testing before integration

Individual components must pass their tests before being integrated into the complete system.

### Rule 7 — Numerical correctness

Gradient calculations must be checked against numerical approximations.

### Rule 8 — Data integrity

The test dataset must never be used to optimize model parameters.

### Rule 9 — No unnecessary complexity

Do not introduce GPUs, distributed computing, databases or complex infrastructure unless a demonstrated requirement justifies them.

### Rule 10 — Documentation

All important modules, configurations and mathematical functions must be documented.

### Rule 11 — Dependency control

Every external dependency must have a documented purpose. No unnecessary frameworks.

### Rule 12 — Security

Validate incoming image data, limit request sizes, avoid unsafe deserialization and do not expose model files or arbitrary server-side file access through the API.

# 7. Testing and acceptance criteria

The project cannot be declared complete merely because it predicts a few handwritten numbers correctly.

| Test             | Expected result                                           |
| ---------------- | --------------------------------------------------------- |
| ReLU test        | Correct activation values                                 |
| Softmax test     | Scores sum to approximately 1                             |
| Loss test        | Correct cross-entropy                                     |
| Gradient test    | Analytical and numerical gradients agree within tolerance |
| Training test    | Loss decreases on a small dataset                         |
| Model save/load  | Predictions remain consistent                             |
| Dataset test     | Only labels 1–9 remain                                    |
| Blank canvas     | Rejected                                                  |
| Input dimensions | Exactly 784 features                                      |
| Integration test | Canvas prediction works                                   |
| Offline test     | Prediction works without internet                         |

### Accuracy evaluation

We will track:

- Overall test accuracy.
- Per-digit precision.
- Per-digit recall.
- F1 score.
- Confusion matrix.
- Training loss.
- Validation loss.
- Inference latency.

A 97% test accuracy target is the initial objective, not a release requirement that can be claimed without evidence.

# 8. Development roadmap

## Project milestones

1. M1 — Mathematical foundation

   Implement neurons, weights, biases, activation functions and matrix operations.
2. M2 — Neural network engine

   Implement forward propagation, loss, backpropagation and optimizer.
3. M3 — Training pipeline

   Load MNIST, filter labels, train and record experiments.
4. M4 — Evaluation

   Test accuracy, gradient correctness, generalization and model reliability.
5. M5 — Inference engine

   Save and load the trained model independently of the training process.
6. M6 — Application development

   Build the drawing interface and Flask integration.
7. M7 — Final validation

   Test offline operation, different handwriting styles and edge cases.
8. M8 — Release 1.0

   Documentation, packaging, release notes and reproducible installation.

# 9. Release management and future expansion

## Version roadmap

| Version | Planned capabilities                    |
| ------- | --------------------------------------- |
| V1.0    | Handwritten digits 1–9                  |
| V1.1    | Improved preprocessing and robustness   |
| V1.2    | Training experiment dashboard           |
| V1.3    | User-selectable training configurations |
| V2.0    | Recognition of digits 0–9               |
| V2.1    | Additional handwriting datasets         |
| V3.0    | Experimental CNN architecture           |

Future versions must not compromise the simplicity and transparency of V1.0.

## 9.1 Final deliverables

- Independently implemented neural network.
- Custom backpropagation engine.
- Custom optimizer.
- MNIST training pipeline.
- Trained model.
- Model evaluation report.
- Interactive drawing application.
- Prediction API.
- Unit and integration tests.
- PRD, TRD and engineering rules.
- Mathematical documentation.
- Installation and usage documentation.
- Versioned release.

# 10. Definition of project completion

Synapse V1.0 will be considered complete when:

1. The neural network is implemented without a prebuilt training framework.
2. The mathematical implementation passes its tests.
3. The model successfully trains on MNIST digits 1–9.
4. The model achieves its documented evaluation results.
5. Users can draw digits through the interface.
6. Predictions are generated using the trained model.
7. The application operates locally without an internet connection.
8. The project can be installed and reproduced using its documentation.

# 11. Master project specification

Here's the consolidated project brief in a document you can keep as the project's initial specification.

# Shadowcr3 Synapse

## Product & Technical Specification — V1.0

**Project:** Shadowcr3 Synapse\
**Category:** Artificial Intelligence / Machine Learning\
**Status:** Proposed\
**Specification date:** November 2026\
**Primary language:** Python

### 1. Project overview

Shadowcr3 Synapse is a lightweight handwritten digit recognition system powered by an independently implemented artificial neural network.

The application accepts handwritten digits from 1 to 9 through a digital drawing canvas, preprocesses the input, and predicts the most likely digit.

The primary objective is to understand and implement the mathematical foundations of neural networks rather than depend on pretrained neural network frameworks.

### 2. Primary objectives

- Implement a feedforward neural network from scratch.
- Implement forward propagation and backpropagation.
- Implement activation and loss functions.
- Implement an optimizer.
- Train the network using MNIST.
- Recognize handwritten digits 1–9.
- Develop an interactive drawing interface.
- Integrate the neural network with a local application.
- Evaluate and document model performance.

### 3. Initial architecture

| Layer    | Neurons | Activation |
| -------- | ------- | ---------- |
| Input    | 784     | None       |
| Hidden 1 | 128     | ReLU       |
| Hidden 2 | 64      | ReLU       |
| Output   | 9       | Softmax    |

Total trainable parameters: 109,385.

### 4. Technology stack

- Python 3
- NumPy
- Matplotlib
- Flask
- HTML5
- CSS3
- JavaScript
- MNIST dataset
- JSON configuration
- NumPy NPZ model storage

### 5. Implementation restrictions

The neural network's training and inference mathematics must be implemented independently.

Prohibited in the core neural network:

- PyTorch
- TensorFlow
- Keras
- Prebuilt neural network training engines

NumPy is permitted for numerical and matrix operations.

### 6. Training configuration

Initial experimental configuration:

- Dataset: MNIST, filtered to digits 1–9.
- Input: 28×28 grayscale.
- Epochs: 20.
- Batch size: 64.
- Learning rate: 0.001.
- Optimizer: Custom Adam.
- Loss: Cross-entropy.
- Random seed: 42.

### 7. Performance objectives

- Target test accuracy: 97% or higher.
- Target inference latency: below 100 ms.
- Target model size: below 5 MB.
- Local offline inference.
- Reproducible training and evaluation.

All performance figures must be measured and documented.

### 8. Functional requirements

1. Accept handwritten drawings.
2. Clear and reset the canvas.
3. Preprocess input images.
4. Convert images to 784 numerical features.
5. Predict one of nine classes.
6. Display the predicted digit.
7. Display all nine output scores.
8. Load previously trained models.
9. Display training and evaluation metrics.
10. Handle blank and invalid input.

### 9. Engineering rules

1. No black-box neural network training frameworks.
2. All learning algorithms must be independently implemented.
3. Mathematical correctness takes priority over performance optimization.
4. Test every core mathematical component.
5. Verify gradients numerically.
6. Preserve the independent test dataset.
7. Keep modules independent and maintainable.
8. Record all training configurations.
9. Never fabricate performance results.
10. Document every major implementation decision.
11. Validate external inputs.
12. Avoid unnecessary dependencies and infrastructure.

### 10. Application design

A premium dark interface with graphite backgrounds, electric-blue accents, cyan highlights and minimal typography.

Primary interface components:

- Drawing canvas.
- Predict button.
- Clear button.
- Prediction display.
- Nine-class output visualization.
- Model status.
- Inference latency.
- Training metrics dashboard.

### 11. Development milestones

M1 — Mathematical foundation\
M2 — Neural network implementation\
M3 — Training pipeline\
M4 — Model evaluation\
M5 — Inference implementation\
M6 — Drawing interface\
M7 — Integration and testing\
M8 — Documentation and release

### 12. Final goals

The final product must be a functional, locally executable handwritten digit recognition application.

The developer must be able to explain how the network processes inputs, calculates errors, propagates gradients and updates its parameters.

The project must demonstrate actual learning rather than simply return hardcoded predictions.

### 13. Future development

- Improved handwriting preprocessing.
- Expanded digit recognition.
- Experimental CNN architecture.
- Additional datasets.
- Training visualization.
- Configurable experiments.

**Project philosophy: Understand the mathematics. Build the engine. Train the model. Verify the results.**

## 12.recommended development approach

Don't rush directly into the UI. The most valuable part of this project is the neural network itself.

develop it in this order:

1. Make a single neuron work.
2. Make multiple neurons work together.
3. Implement a complete feedforward network.
4. Make it learn a tiny synthetic dataset.
5. Verify backpropagation mathematically.
6. Only then introduce MNIST.