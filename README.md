# MOS

## Setting up the Environment
1. Create a conda environment:
```
conda create -n mos python=3.10 pandas seaborn librosa onnxruntime-gpu fairseq
```
2. Activate the conda environment:
```
conda activate mos
```

## Running Inference
1. Navigate to the "_mos" directory:
```
cd _mos
```
2. Execute the wav_to_csv.py script for inference:
```
python wav_to_csv.py
```
