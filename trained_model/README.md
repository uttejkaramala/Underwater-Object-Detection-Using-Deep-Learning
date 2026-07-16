# TensorFlow Dehazing Model

This directory stores the TensorFlow SavedModel used for underwater image dehazing.

Expected structure:

```text
trained_model/
├── saved_model.pb
└── variables/
    ├── variables.data-00000-of-00001
    └── variables.index
```

The trained model files are not included in this repository due to their size.

Place the SavedModel in this folder before running the application.