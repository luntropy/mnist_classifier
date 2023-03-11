"""Model config in json format"""

CFG = {
    "data": {
        "image_shape": [28, 28, 1]
    },
    "train": {
        "model_save_path": "saved_models",
        "batch_size": 64,
        "buffer_size": 10000,
        "validation_split": 0.1,
        "epoches": 3,
        "learning_rate": 0.001,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input_shape": [28, 28, 1],
        "output_shape": 10,
        "conv2d_units": 32,
        "conv2d_kernel": 3,
        "conv2d_activation": "relu",
        "dense_units": 128,
        "dense_activation": 'relu'
    }
}
