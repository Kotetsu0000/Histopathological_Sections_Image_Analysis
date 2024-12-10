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
|   │  ├─y_membrane     (最初から入れておく必要あり→[ans_thin.png], 実験開始時に作成→[ans.png])
│   │  └─y_nuclear      (最初から入れておく必要あり→[ans.png], 実験開始時に作成→[red.png, green.png])
│   ├─02
│   │  ├─x              (最初から入れておく必要あり→[bf.png, df.png, he.png])
|   │  ├─y_membrane     (最初から入れておく必要あり→[ans_thin.png], 実験開始時に作成→[ans.png])
│   │  └─y_nuclear      (最初から入れておく必要あり→[ans.png], 実験開始時に作成→[red.png, green.png])
│   同一標本内の画像の枚数分続く
├─pathological_specimen_02
│   ├─01
|   ├─02
|   同一標本内の画像の枚数分続く
標本数分続く
```

## 環境構築方法

※Windowsの場合はVisualstudioのC++によるデスクトップ開発を事前にインストールしておく必要があります。
※Docker、Linux等はGCCがインストールされていれば問題ないはずですが、現在はWindowsのみでの動作確認を行っています。

### 1. PyTorchのインストール

CUDAのインストールはPyTorch2.0以降不要になったため、NVIDIAのGPUドライバのみインストールしてください。

PyTorchのインストール方法は[公式サイト](https://pytorch.org/get-started/locally/)を参照してください。このコードはCUDAを使用することを前提としており、CUDAが使用できない場合はコードの一部を変更する必要があります。

### 2. その他のライブラリのインストール

その後、以下のコマンドを実行して必要なライブラリをインストールしてください。

```bash
pip install git+https://github.com/Kotetsu0000/VitLib.git git+https://github.com/Kotetsu0000/VitLib_PyTorch.git
```


