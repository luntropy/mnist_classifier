#!/bin/bash

docker run --rm --gpus all -e API_PORT=$API_PORT -p $API_PORT:$API_PORT mnist-classifier-app
