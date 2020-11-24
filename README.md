## DataRobot MLOps Git Integration

## Test Insurance LGBM Model with Shap model via [DRUM](https://github.com/datarobot/datarobot-user-models)

Structured 
```
pip install -U datarobot-drum
pip install -r ./insurance-lgbm-shap/requirements.txt
drum score --code-dir ./insurance-lgbm-shap --input ./data/test_data.csv --target-type regression --verbose
```

Try it out in unstructured mode and get shap explanations
```
drum server --code-dir ./model-v2 --input ./extras/test_data/test_data.csv --target-type unstructured --verbose
```

## Test out IMDB Graph Isomorphism Network via [DRUM](https://github.com/datarobot/datarobot-user-models)

```
pip install -U datarobot-drum
pip install -r ./imdb-gin/requirements.txt
drum score --code-dir ./imdb-gin --input ./data/imdb-graph.avro --target-type unstructured --content-type application/octet-stream
```

See the Unstructured Model Predict
