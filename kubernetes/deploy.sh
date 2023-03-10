#!/bin/bash

kubectl apply -f flask_deployment.yaml
kubectl apply -f flask_service.yaml
kubectl apply -f flask_ingress.yaml

minikube addons enable ingress

