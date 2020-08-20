#!/bin/sh
cd /opt/code/ || exit 1
export PYTHONPATH=/opt/code
#export FLASK_APP=server.app
#python -m flask run --host=0.0.0.0 --port 8080

CMD="cmrun -m . --server 0.0.0.0:8080"

if [ ! -z "${POSITIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --positive-class-label ${POSITIVE_CLASS_LABEL}"
fi
if [ ! -z "${NEGATIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --negative-class-label ${NEGATIVE_CLASS_LABEL}"
fi

${CMD}