# MOS
This MOS (Mean Opinion Score) system integrates components from various sources:
1. DNSMOS (https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS)
2. NISQA (https://github.com/gabrielmittag/NISQA)
3. MOSSSL (https://github.com/nii-yamagishilab/mos-finetune-ssl)
4. SIGMOS (https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2024/sigmos)

For each audio file, the system utilizes the librosa library to read it as a waveform.
Subsequently, it performs inference for four MOS metrics, generating 20 scores in total, and then saves the results in a CSV file.

## Setting up the Environment
1. Create a conda environment:
```
conda create -n mos python=3.10 
```
2. Activate the conda environment:
```
conda activate mos
```
3. Install required packages using pip:
```
pip install pandas seaborn librosa onnxruntime-gpu fairseq tensorboardX
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
