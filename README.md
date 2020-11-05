## DataRobot MLOps Git Integration

## Test model via [DRUM](https://github.com/datarobot/datarobot-user-models)


`pip install -U datarobot-drum`

```
drum score --code-dir ./model --input ./extras/test_data/test_data.csv --target-type regression --verbose
```