#!/usr/bin/env bash

docker rmi torch_tools:latest
docker build -t torch_tools:latest .
