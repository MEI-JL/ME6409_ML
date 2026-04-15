Heavily based on provided notebook in ME6409 course in Spring 2026.
## Instructions
### Preprocessing
Unzip `ProcessedData.zip` in the same directory and run `trim_dataset.py`.
### Run on colab (prefered)
Zip `/ProcessedDataTrimmed`, upload it and `walkthrough_colab.ipynb` to your google drive, open the notebook and follow the instructions.
### Run Locally
Set up the environment and follow `walkthrough.ipynb`.

## Local env setup
You need to change the torch version in `requirements.txt`.
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```
