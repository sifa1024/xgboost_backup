#!/usr/bin/env bash


docker build -t sifa1024/xgboostserver:v0.2.0 -f Dockerfile .
docker push sifa1024/xgboostserver:v0.2.0
