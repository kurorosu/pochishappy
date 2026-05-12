# 変更履歴

このプロジェクトの主な変更はすべてこのファイルに記録します。

フォーマットは [Keep a Changelog](https://keepachangelog.com/) に準拠し、
バージョニングは [Semantic Versioning](https://semver.org/) に従います。

## [Unreleased]

### Added

- GitHub Issue テンプレート (bug, feature, docs, refactor, test) と PR テンプレートを追加し, あわせて CHANGELOG.md と changelogs/ アーカイブディレクトリを初期化. ([#1](https://github.com/kurorosu/pochishappy/pull/1))
- pre-commit / uv を導入し Python 3.14 環境を構築. pochishappy パッケージのスケルトンと最小テストを追加. ([#2](https://github.com/kurorosu/pochishappy/pull/2))
- Codex 向け AGENTS.md をローカル専用ファイルとして ignore 対象に追加. ([#3](https://github.com/kurorosu/pochishappy/pull/3))
- `pochi` CLI の骨格実装, SHAP 実行パイプライン雛形, pochitrain 互換 logger, CLI テストを追加. ([#4](https://github.com/kurorosu/pochishappy/pull/4))
- `.codex` ディレクトリを `.gitignore` に追加し, ローカル設定を追跡対象外に変更. ([#5](https://github.com/kurorosu/pochishappy/pull/5))
- `pochi` CLI を実モデルで動作可能にする SHAP 実行ロジック (`GradientExplainer`, `shap.image_plot`) を実装. 学習プロジェクト固有のパラメータを JSON config で受け取る `pochishappy.config` モジュールを追加し, torch / torchvision (CUDA 13.0) / matplotlib を依存に追加. pytest slow marker で SHAP 計算テストを既定除外. ([#6](https://github.com/kurorosu/pochishappy/pull/6))
- `configs/example.json` に ImageNet 既定値の動作するサンプル config を同梱. README にコピー → 書き換えのフローと, `input_size` / `resize` の関係 (torchvision `Resize` の挙動差) を追記. (NA.)

### Changed

- 無し

### Fixed

- 無し

### Removed

- 無し

## 過去の変更履歴

古いバージョンの履歴は [`changelogs/`](changelogs/) ディレクトリにアーカイブされています。
