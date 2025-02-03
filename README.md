# Histopathological_Sections_Image_Analysis

肝病理組織画像から細胞膜、細胞核のセグメンテーションを行う際の最適な撮像法に関する実験を行うための実験ソースコードです。

## データセット

データセットは光学顕微鏡にて撮影された「明視野撮像法」「暗視野撮像法」「位相差撮像法」の画像を使用します。ひと組みの画像に対して、細胞膜、細胞核のセグメンテーションを行うためのセグメンテーション画像を用意する必要があります。

### データセットのフォルダ構成

```
**フォルダ構成**
├─pathological_specimen_01(同一標本をまとめるフォルダ, フォルダ名は任意で可)
│   ├─01(フォルダ名は任意で可)
│   │  ├─x              (最初から入れておく必要あり→[bf.png, df.png, he.png])
|   │  ├─y_membrane     (最初から入れておく必要あり→[ans_thin.png], 実験開始時に作成→[ans.png, ans_nograd.png])
│   │  └─y_nuclear      (最初から入れておく必要あり→[ans.png], 実験開始時に作成→[red.png, green.png])
│   ├─02
│   │  ├─x              (最初から入れておく必要あり→[bf.png, df.png, he.png])
|   │  ├─y_membrane     (最初から入れておく必要あり→[ans_thin.png], 実験開始時に作成→[ans.png, ans_nograd.png])
│   │  └─y_nuclear      (最初から入れておく必要あり→[ans.png], 実験開始時に作成→[red.png, green.png])
│   同一標本内の画像の枚数分続く
├─pathological_specimen_02
│   ├─01
|   ├─02
|   同一標本内の画像の枚数分続く
標本数分続く
```

- x: 光学顕微鏡にて撮影された画像
  - bf.png: 明視野撮像法による画像
  - df.png: 暗視野撮像法による画像
  - he.png: 位相差撮像法による画像
- y_membrane: 細胞膜のセグメンテーション画像
    - ans_thin.png: 細胞膜のセグメンテーション画像(細線化されている)
    - ans.png: 細胞膜のセグメンテーション画像(グラデーションありの膨張画像)
    - ans_nograd.png: 細胞膜のセグメンテーション画像(グラデーションなしの膨張画像)
- y_nuclear: 細胞核のセグメンテーション画像
    - ans.png: 細胞核のセグメンテーション画像
    - red.png: 細胞核のセグメンテーション画像(考慮外核)
    - green.png: 細胞核のセグメンテーション画像(考慮外核以外の核)
    - eval.png: 明視野撮像法と細胞核の正解(考慮外核: 赤, 考慮外核以外: 緑)を重ねた画像

## 環境構築方法(物理マシンに環境構築する場合)

※Windowsの場合はVisualstudioのC++によるデスクトップ開発を事前にインストールしておく必要があります。
※Docker、Linux等はGCCがインストールされていれば問題ないはずですが、現在はWindowsのみでの動作確認を行っています。

### 0. 事前準備

Pythonのバージョンは3.11での動作確認を行っています。Pythonのバージョンが異なる場合は動作しない可能性があります。

### 1. リポジトリのクローン

まず、このリポジトリをクローンしてください。

```bash
git clone https://github.com/Kotetsu0000/Histopathological_Sections_Image_Analysis.git
cd Histopathological_Sections_Image_Analysis
```

必要があれば、仮想環境を作成してください。(以下はWindowsのコマンド例)

```bash
python -m venv venv
.\venv\Scripts\activate
```

Linuxの場合は以下のコマンドを実行してください。

```bash
python -m venv venv
source venv/bin/activate
```

### 2. PyTorchのインストール

CUDAのインストールはPyTorch2.0以降不要になったため、NVIDIAのGPUドライバのみインストールしてください。

PyTorchのインストール方法は[公式サイト](https://pytorch.org/get-started/locally/)を参照してください。このコードはCUDAを使用することを前提としており、CUDAが使用できない場合はコードの一部を変更する必要があります。

本実験ではPyTorchのバージョンは2.5.1での動作確認を行っています。

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. その他のライブラリのインストール

その後、以下のコマンドを実行して必要なライブラリをインストールしてください。

```bash
pip install git+https://github.com/Kotetsu0000/VitLib.git git+https://github.com/Kotetsu0000/VitLib_PyTorch.git pandas
```

### 4. 実験の実行

まず、`config.ini`ファイルを作成し、`experiment.py`を実行してください。[こちら](config/config.ini)に記載例があります。

```bash
python experiment.py --config config.ini
```

> [!IMPORTANT]
> configオプションを指定しない場合は、すべてデフォルトの設定で実行されます。

#### config.iniの設定

config.iniの設定は以下の通りです。

| 設定項目 | 説明 |
| --- | --- |
|img_path|1600×1200の画像の入ったディレクトリ|
|default_path|結果を保存するディレクトリ|
|experiment_subject|実験対象(membrane or nuclear or both)|
|use_network|使用するネットワーク(U-Net or U-Net++)|
|deep_supervision|Deep Supervisionを使用するか(True or False, U-Net++のみ)|
|color|画像の色空間(RGB or HSV)|
|blend|ブレンドモード(concatenate or alpha)|
|gradation|グラデーションを使用するか(True or False, membrane or bothの場合のみ)|
|train_dont_care|考慮外核を学習するか(True or False, nuclear or bothの場合のみ)|
|care_rate|標準的核面積に対する考慮外核の面積の割合(0~100, 単位[%], nuclear or bothの場合のみ)|
|lower_ratio|標準的核面積の計算時に除外する面積下位割合(0~100, 単位[%], nuclear or bothの場合のみ)|
|higher_ratio|標準的核面積の計算時に除外する面積上位割合(0~100, 単位[%], nuclear or bothの場合のみ)|
|use_loss|損失関数(DiceLoss or BCELoss)|
|start_num|学習を開始するエポック数|
|num_epochs|エポック数|
|lr|学習率|
|batch_size|バッチサイズ|
|use_list_length|撮像法の組み合わせの実験なら3, 撮像法の全チャンネル(9チャンネル)での実験は9(未実装), RGB, HSVを組み合わせて全チャンネルの比較実験を行う場合は18(未実装)|
|data_augmentation_num|データ拡張の回数|
|train_size|学習画像のサイズ|
|saturation_mag|彩度の変化の範囲|
|value_mag|明度の変化の範囲|
|contrast_mag|コントラストの変化の範囲|
|radius_train|学習時の膨張の半径(membrane or bothの場合のみ)|
|use_device|使用するデバイス(CUDAのナンバーのリスト)|
|use_autocast|自動混合精度を使用するか(True or False)|
|autocast_dtype|自動混合精度のデータ型(bfloat16 or float16)|
|compress_rate|画像の圧縮率(1で圧縮なし, 同一画素値の圧縮幅)|
|ignore_error|エラーを無視するか(True or False)|
|radius_eval|評価時の膨張の半径(membrane or bothの場合のみ)|
|membrane_sparse_epoch_start|細胞膜の評価を開始するEpoch(membrane or bothの場合のみ)|
|membrane_sparse_epoch_step|細胞膜の評価を行うEpochのステップ(membrane or bothの場合のみ)|
|membrane_sparse_threshold_min|細胞膜の評価の二値化閾値の最小値(membrane or bothの場合のみ)|
|membrane_sparse_threshold_max|細胞膜の評価の二値化閾値の最大値(membrane or bothの場合のみ)|
|membrane_sparse_threshold_step|細胞膜の評価の閾値のステップ(membrane or bothの場合のみ)|
|membrane_sparse_del_area_min|細胞膜の評価の小領域削除面積の最小値(membrane or bothの場合のみ)|
|membrane_sparse_del_area_max|細胞膜の評価の小領域削除面積の最大値(membrane or bothの場合のみ)|
|membrane_sparse_del_area_step|細胞膜の評価の小領域削除面積のステップ(membrane or bothの場合のみ)|
|nuclear_eval_mode|核の評価を開始するEpoch(nuclear or bothの場合のみ)|
|nuclear_eval_distance|核の評価を行うEpochのステップ(nuclear or bothの場合のみ)|
|nuclear_sparse_epoch_start|核の評価の二値化閾値の最小値(nuclear or bothの場合のみ)|
|nuclear_sparse_epoch_step|核の評価の二値化閾値の最小値(nuclear or bothの場合のみ)|
|nuclear_sparse_threshold_min|核の評価の二値化閾値の最小値(nuclear or bothの場合のみ)|
|nuclear_sparse_threshold_max|核の評価の二値化閾値の最大値(nuclear or bothの場合のみ)|
|nuclear_sparse_threshold_step|核の評価の二値化閾値のステップ(nuclear or bothの場合のみ)|
|nuclear_sparse_del_area_min|核の評価の小領域削除面積の最小値(nuclear or bothの場合のみ)|
|nuclear_sparse_del_area_max|核の評価の小領域削除面積の最大値(nuclear or bothの場合のみ)|
|nuclear_sparse_del_area_step|核の評価の小領域削除面積のステップ(nuclear or bothの場合のみ)|

### 5. 評価の実行

実験で使用したものと同じ`config.ini`ファイルを使用して、`evaluation.py`を実行してください。

```bash
python evaluation.py --config config.ini
```

> [!IMPORTANT]
> configオプションを指定しない場合は、すべてデフォルトの設定で実行されます。

プログラムが終了すると、`{{default_path}}/log_membrane/test/all_aggregate.csv`または`{{default_path}}/log_nuclear/test/all_aggregate.csv`に評価結果が保存されます。
