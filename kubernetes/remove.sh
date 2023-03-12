#!/bin/bash

kubectl delete deployment mnist-classifier-app
kubectl delete service mnist-classifier-app-service
kubectl delete ingress mnist-classifier-app-ingress
