# ベースイメージを指定
FROM nvcr.io/nvidia/pytorch:25.01-py3

# 作業ディレクトリを設定（必要に応じて変更してください）
WORKDIR /app

# git と必要なライブラリをインストール
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && pip install --upgrade pip\
    && pip install git+https://github.com/Kotetsu0000/VitLib.git git+https://github.com/Kotetsu0000/VitLib_PyTorch.git pandas \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
