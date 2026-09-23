## Dataset Setup

The datasets are not included in this repository because of their large size.

### 1. Multi-Disease Dataset

Download the four-class lung disease dataset used for multi-class classification.

Place the extracted dataset at:

```text
FedLung-XAI/data/multidisease/
```

Maintain the class folders required by the data loader.

### 2. Chest X-ray Dataset

The NORMAL and PNEUMONIA dataset is used for the binary classification experiments.

Place the extracted dataset at:

```text
FedLung-XAI/data/chest_xray/
```

Maintain the expected train, validation, and test folder structure.

### Final Directory Structure

```text
FedLung-XAI/
├── backend/
├── frontend/
├── src/
├── data/
│   ├── multidisease/
│   └── chest_xray/
├── requirements.txt
└── README.md
```

**Important:** Download the datasets from their original sources and follow their licensing requirements. The dataset files themselves are not included in this repository.
