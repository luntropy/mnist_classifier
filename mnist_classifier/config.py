"""Model config in json format"""

CFG = {
    "data": {
        "image_height": 28,
        "image_width": 28,
        "image_channels": 1
    },
    "train": {
        "model_save_path": "saved_models",
        "batch_size": 64,
        "buffer_size": 10000,
        "epoches": 3,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input_shape": [28, 28, 1],
        "output_shape": 10
    }
}
