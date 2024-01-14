from DNSMOS import dnsmos_local  # 16_000
from NISQA_master.nisqa.NISQA_model import nisqaModel
from mos_finetune_ssl_main import predict_noGT  # 16_000
from sigmos.sigmos import SigMOS  # 48_000
import librosa
import argparse
import os
import glob
import pandas as pd

def process_waveform(waveform, sampling_rate):
    # DNSMOS
    args_dnsmos = argparse.Namespace(
        testset_dir=waveform,
        csv_path="",
        personalized_MOS=True,
        sampling_rate=sampling_rate,
    )
    df_dnsmos = dnsmos_local.main(args_dnsmos).drop(
        ["filename", "sr", "num_hops"], axis=1
    )

    # NISQA
    args_nisqa = argparse.Namespace(
        mode="predict_file",
        pretrained_model="./NISQA_master/weights/nisqa.tar",
        deg=waveform,
        ms_channel=1,
        output_dir=None,
    )
    df_nisqa = nisqaModel(vars(args_nisqa)).predict().drop(["deg"], axis=1)

    # mosssl
    args_mosssl = argparse.Namespace(
        fairseq_base_model="./mos_finetune_ssl_main/fairseq/wav2vec_small.pt",
        datadir=waveform,
        finetuned_checkpoint="./mos_finetune_ssl_main/pretrained/ckpt_w2vsmall",
        outfile="./mos_finetune_ssl_main/answer.txt",
    )
    df_mosssl = predict_noGT.main(args_mosssl)

    # sigmos
    sigmos_estimator = SigMOS(model_dir="sigmos")
    df_sigmos = pd.DataFrame(
        columns=[
            "MOS_COL",
            "MOS_DISC",
            "MOS_LOUD",
            "MOS_NOISE",
            "MOS_REVERB",
            "MOS_SIG",
            "MOS_OVRL",
        ]
    )
    sigmos_results = sigmos_estimator.run(waveform, sr=sampling_rate)
    df_sigmos = df_sigmos._append(
        pd.Series(
            {
                "MOS_COL": sigmos_results["MOS_COL"],
                "MOS_DISC": sigmos_results["MOS_DISC"],
                "MOS_LOUD": sigmos_results["MOS_LOUD"],
                "MOS_NOISE": sigmos_results["MOS_NOISE"],
                "MOS_REVERB": sigmos_results["MOS_REVERB"],
                "MOS_SIG": sigmos_results["MOS_SIG"],
                "MOS_OVRL": sigmos_results["MOS_OVRL"],
            }
        ),
        ignore_index=True,
    )

    # merge
    df = df_dnsmos
    df = pd.concat([df, df_nisqa], axis=1)
    df["MOS_SSL"] = df_mosssl
    df = pd.concat([df, df_sigmos], axis=1)

    print(f'[len_in_sec] = {df["len_in_sec"][0]}') # not used
    print(f'OVRL_raw = {df["OVRL_raw"][0]}')
    print(f'SIG_raw = {df["SIG_raw"][0]}')
    print(f'BAK_raw = {df["BAK_raw"][0]}')
    print(f'OVRL = {df["OVRL"][0]}')
    print(f'SIG = {df["SIG"][0]}')
    print(f'BAK = {df["BAK"][0]}')
    print(f'P808_MOS = {df["P808_MOS"][0]}')
    print(f'mos_pred = {df["mos_pred"][0]}')
    print(f'noi_pred = {df["noi_pred"][0]}')
    print(f'dis_pred = {df["dis_pred"][0]}')
    print(f'col_pred = {df["col_pred"][0]}')
    print(f'loud_pred = {df["loud_pred"][0]}')
    print(f'MOS_SSL = {df["MOS_SSL"][0]}')
    print(f'MOS_COL = {df["MOS_COL"][0]}')
    print(f'MOS_DISC = {df["MOS_DISC"][0]}')
    print(f'MOS_LOUD = {df["MOS_LOUD"][0]}')
    print(f'MOS_NOISE = {df["MOS_NOISE"][0]}')
    print(f'MOS_REVERB = {df["MOS_REVERB"][0]}')
    print(f'MOS_SIG = {df["MOS_SIG"][0]}')
    print(f'MOS_OVRL = {df["MOS_OVRL"][0]}')
    print(f'[mean MOS] = {df.drop(["len_in_sec"], axis=1).iloc[0].mean(axis=0)}') # not used
    return df

if __name__ == "__main__":
    wavdir = "/home/alexis/_MixedDataset/train_clean"
    df = pd.DataFrame(
        columns=[
            "filename",
            "len_in_sec", # not used
            "OVRL_raw",
            "SIG_raw",
            "BAK_raw",
            "OVRL",
            "SIG",
            "BAK",
            "P808_MOS",
            "mos_pred",
            "noi_pred",
            "dis_pred",
            "col_pred",
            "loud_pred",
            "MOS_SSL",
            "MOS_COL",
            "MOS_DISC",
            "MOS_LOUD",
            "MOS_NOISE",
            "MOS_REVERB",
            "MOS_SIG",
            "MOS_OVRL",
        ]
    )
    count = 0
    for wav in glob.glob(os.path.join(wavdir, "*.wav")):
        sampling_rate = 16_000
        waveform = librosa.load(wav, sr=sampling_rate)[0]
        print(f"========== Processing Waveform No.{count} [{wav}] ==========")
        df_wav = process_waveform(waveform, sampling_rate)
        df_wav["filename"] = wav
        df = df._append(df_wav, ignore_index=True)
        count += 1

    df.to_csv("df_train_clean.csv", index=False)