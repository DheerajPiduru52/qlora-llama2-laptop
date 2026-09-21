#!/bin/sh
# SageMaker starts the container as `docker run <image> serve`. Map that to the full serve command so the
# container defaults (0.0.0.0:8080, merged model at /opt/ml/model) survive the CMD being replaced.
set -e
if [ "$1" = "serve" ] || [ "$#" -eq 0 ]; then
  [ "$#" -gt 0 ] && shift
  exec qlora serve -c serve_container.yaml "$@"
fi
exec qlora "$@"
