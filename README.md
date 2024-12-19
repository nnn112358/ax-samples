[英語版](./README_EN.md) | [简体中文](./README_ch.md)|日本語 

# AX-Samples
[![ライセンス](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://raw.githubusercontent.com/AXERA-TECH/ax-samples/main/LICENSE)

| プラットフォーム | ビルドステータス |
| -------- | ------------ |
| AX650N   | ![GitHub Actions ワークフローステータス](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-samples/build_650.yml)|
| AX630C   | ![GitHub Actions ワークフローステータス](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-samples/build_630c_glibc.yaml)|
| AX620Q   | ![GitHub Actions ワークフローステータス](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-samples/build_620q_uclibc.yaml)|
| AX620A   | ![GitHub Actions ワークフローステータス](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-samples/build_620a.yml)|

## はじめに
**AX-Samples**は**[AXERA-TECH](https://www.axera-tech.com/)**が主導して開発しています。このプロジェクトは、一般的な**オープンソース深層学習アルゴリズム**の**AXERA-TECH**の**AI SoC**上での実装例を提供し、コミュニティの開発者が迅速な評価と適応を行えるようにすることを目的としています。

対応チップ:
- [AX630C](docs/AX630C.md)/[AX620Q](docs/AX620Q.md)
- [AX650A](docs/AX650A.md)/[AX650N](docs/AX650N.md)
- [AX620A](docs/AX620A.md)/[AX620U](docs/AX620U.md)
- [AX630A](docs/AX630A.md)

対応開発ボード:
- [AXera-Pi](https://wiki.sipeed.com/m3axpi)(AX620A)
- [AXera-Pi Pro](https://wiki.sipeed.com/m4ndock)(AX650N)
- [AXera-Pi Zero](https://axera-pi-zero-docs-cn.readthedocs.io/zh-cn/latest/index.html)(AX620Q)

## クイックスタート
### ビルド
- [クイックビルド](docs/compile.md) cmakeを使用したシンプルなクロスプラットフォームビルドを実現。

### サンプル
- [examples](examples/) 一般的な分類、検出、姿勢などの深層学習オープンソースアルゴリズムと従来のCV操作の使用例を提供。issueの要望に応じて継続的に更新。

### クラウドリソース
- **ModelZoo**、**プリコンパイルプログラム**、**テスト画像**などのコンテンツを提供:
  - [Baidu Cloud](https://pan.baidu.com/s/1cnMeqsD-hErlRZlBDDvuoA?pwd=oey4)
  - [Google Drive](https://drive.google.com/drive/folders/1JY59vOFS2qxI8TkVIZ0pHfxHMfKPW5PS?usp=sharing)

### パフォーマンス評価
- [Benchmark](benchmark/) 一般的なオープンソースモデルの推論時間統計。*AXera-Pi*、*AXera-Pi Pro*、*AXera-Pi Zero*での実測に基づく。

## 関連プロジェクト
- NPUツールチェーンのオンラインドキュメント。NPUツールチェーンの使用説明と入手方法を提供:
  - [Pulsar](https://pulsar-docs.readthedocs.io/zh_CN/latest/)(AX630A/AX620A/AX620U対応)
  - [Pulsar2](https://pulsar2-docs.readthedocs.io/zh_CN/latest/)(AX650A/AX650N/AX630C/AX620Q対応)

## 技術ディスカッション
- Github issues
- QQグループ: 139953715
