# LLM Visualizer Streamlit

**LLM Visualizer Streamlit** は、大規模言語モデル（LLM）の生成挙動に影響を与える各種パラメータ（Temperature / Top-p / Top-k）を調整し、その出力結果や注意機構（Attention）を視覚的に観察できるツールです。  
GPU環境でのローカル実行を前提としており、モデルは `gpt2-medium` に限定されています。

## 特徴

- **`gpt2-medium` モデルに特化**  
  Hugging Face の `gpt2-medium` を使用し、他モデルの切り替え機能は搭載していません。

- **出力パラメータの調整**  
  - Temperature  
  - Top-p  
  - Top-k  
  これらをスライダーでインタラクティブに変更可能です。  
  Top-p が 1 の場合は Top-k スライダーが自動的に無効化されるUI制御も実装されています。

- **Attention のヒートマップ可視化**  
  モデルの自己注意（Self-Attention）機構をヒートマップとして可視化し、トークン間の依存関係を直感的に理解できます。

- **チャット形式の出力履歴表示**  
  各出力を時系列で表示し、異なる設定による生成の違いを比較しやすいインターフェースを提供しています。

## 対応環境

- Python 3.10 推奨
- CPU / GPU（CUDA対応）両方で動作可能  
  ※ GPU（NVIDIA）を使用することで高速化が可能です。GPUがない環境では自動的にCPUが使用されます。

⚠️ 注意：
現在の実装は PyTorch の `cuda` デバイスに依存しており、AMD/Intel GPU や CPU 環境では動作しません。

## 想定ユースケース

- LLM の生成挙動を学習・研究目的で観察したい場合
- Temperature / Top-p / Top-k の効果を可視的に比較したい場合
- Attention の可視化を通じてモデルの内部処理を理解したい場合
- 授業やワークショップでのデモンストレーションに使用したい場合

## 今後の展望

- CPU 環境への対応
- `repetition_penalty` パラメータの調整機能の追加

---

※本READMEの文章は OpenAI の ChatGPT-4o によって生成され、内容は開発者によって確認・修正された上で掲載されています。
