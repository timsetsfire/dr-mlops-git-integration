## DataRobot MLOps Git Integration

## Test model via [DRUM](https://github.com/datarobot/datarobot-user-models)

```
pip install -U datarobot-drum
pip install -r ./model-v2/requirements.txt
drum score --code-dir ./model-v2 --input ./extras/test_data/test_data.csv --target-type regression --verbose
```

Try it out in unstructured mode and get shap explanations
```
drum server --code-dir ./model-v2 --target-type unstructured --address localhost:6789 --verbose
```
See the Unstructured Model Predict