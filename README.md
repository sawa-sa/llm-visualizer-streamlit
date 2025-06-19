# LLM Visualizer Streamlit

**LLM Visualizer Streamlit** は、大規模言語モデル（LLM）の生成挙動に影響を与える各種パラメータ（Temperature / Top-p / Top-k）を調整し、その出力結果や注意機構（Attention）を視覚的に観察できるツールです。  
本ツールは **CUDA対応GPU または CPU** 環境でのローカル実行を前提としており、モデルは `gpt2-medium` に限定されています。

---

## 特徴

- ### 🔹 `gpt2-medium` モデル使用  
  Hugging Face の `gpt2-medium` を使用。他モデルの切り替え機能は未対応です。

- ### 🔹 出力パラメータのインタラクティブ調整  
  以下の生成パラメータをリアルタイムで調整可能：
  - `Temperature`
  - `Top-p`
  - `Top-k`  
  ※ Top-p が 1 の場合、自動的に Top-k のUIが無効化されます。

- ### 🔹 Attention のヒートマップ可視化  
  Transformer の Self-Attention をヒートマップとして可視化し、トークン間の依存関係を視覚的に把握可能です。


---

## 対応環境

- Python 3.10 推奨
- 動作環境：**CUDA対応GPU または CPU**
  - GPUが利用可能な場合は自動的に `cuda` を使用
  - GPUがない場合は `cpu` にフォールバック

> ⚠️ **注意：現在の実装は PyTorch の `cuda` デバイスに依存**しています。  
> そのため、**AMD / Intel GPU では動作しません。** CPU 環境は対応済みです。

---

## 想定ユースケース

- LLM の生成挙動を学習・研究目的で観察したい場合
- Temperature / Top-p / Top-k の効果を視覚的に比較したい場合
- Attention の可視化を通じてモデルの挙動を直感的に理解したい場合
- 授業やワークショップでのデモンストレーション

---

## CPU環境で試してみたい方へ（Streamlit Cloud 無料プラン）

以下のリンクから、CPU専用版のデモをブラウザで試すことができます（`gpt2-medium` のみ）：

🔗 **[https://llm-visualizer-app-cpu-fzjyvfhna7sszlugnhuu7u.streamlit.app/](https://llm-visualizer-app-cpu-fzjyvfhna7sszlugnhuu7u.streamlit.app/)**

> ※ 無料プランのため起動に数十秒かかる場合があります。

---

## 今後の展望

- `repetition_penalty` パラメータの調整機能追加
- モデル選択機能の追加（例：ELYZA, GPT-Neo等）

---

> 本READMEの文章は OpenAI の ChatGPT-4o により生成され、開発者によって確認・修正された上で掲載されています。 
