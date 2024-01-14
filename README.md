# MOS
This MOS (Mean Opinion Score) system integrates components from various sources:
1. DNSMOS (https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS): 7 scores
2. NISQA (https://github.com/gabrielmittag/NISQA): 5 scores
3. MOSSSL (https://github.com/nii-yamagishilab/mos-finetune-ssl): 1 score
4. SIGMOS (https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2024/sigmos): 7 scores

For each audio file, the system utilizes the librosa library to read it as a waveform.
Subsequently, it performs inference for four MOS metrics, generating 20 scores in total, and then saves the results in a CSV file.
Note that MOSSSL must run on GPU.

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
1. Navigate to the "MOS" directory:
```
cd MOS
```
2. Execute the wav_to_csv.py script for inference:
```
python wav_to_csv.py
```

## Result
For 1310 test audio files from LJSpeech, we applied various levels of noise at different SNR (Signal-to-Noise Ratio) levels. The distribution of the MOS (Mean Opinion Score) for this batch of audio files is shown in the following chart.
![Alt Text](./mos_distribution.png)
