import numpy as np
import streamlit as st
from config import DEFAULT_PROMPTS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K
from model_loader import load_model
from generator import generate_step
from visualizer import plot_topk, plot_attention

# ─── セッションステートの初期化 ───────────────────────────
state = st.session_state
for key, default in [
    ("input_ids", None),
    ("steps", []),
    ("step_index", 0),
    ("prompt", DEFAULT_PROMPTS[0]),
    ("head_select", "Average"),
    ("lock_params", False)
]:
    if key not in state:
        state[key] = default

# ─── モデルロード ─────────────────────────────────────────
model, tokenizer = load_model()

# ─── UI設定: 探索モードとロック判定 ─────────────────────────
explore_mode = st.checkbox(
    "🔀 探索モード: 途中でパラメータ変更を許可",
    value=False,
    help="オフにすると生成後にパラメータがロックされます"
)
locked = state.lock_params and not explore_mode

st.title("🔍 GPT-2 Medium 可視化デモ")

# ─── プロンプト選択と初期化コールバック ────────────────────
def init_with_template():
    # テンプレートを適用してリセット
    state.prompt = state.prompt_selector
    state.input_ids = tokenizer.encode(state.prompt, return_tensors="pt")
    state.steps = []
    state.step_index = 0
    state.head_select = "Average"
    state.lock_params = False

def init_with_custom():
    # カスタムプロンプトを適用してリセット
    state.prompt = state.prompt_input
    state.input_ids = tokenizer.encode(state.prompt, return_tensors="pt")
    state.steps = []
    state.step_index = 0
    state.head_select = "Average"
    state.lock_params = False

# テンプレート選択ウィジェット
st.selectbox(
    "🧪 プロンプトテンプレート",
    options=DEFAULT_PROMPTS,
    index=DEFAULT_PROMPTS.index(state.prompt) if state.prompt in DEFAULT_PROMPTS else 0,
    key="prompt_selector",
    disabled=locked,
    on_change=init_with_template
)
# カスタム入力ウィジェット
st.text_input(
    "または自分で入力",
    value=state.prompt,
    key="prompt_input",
    disabled=locked,
    on_change=init_with_custom
)

# 初期化ボタン: テンプレート初期化と同じ動作
st.button("🔄 プロンプト初期化", on_click=init_with_template, disabled=False)

# プロンプト未設定なら停止
if state.input_ids is None:
    st.warning("プロンプトを選択または入力して初期化してください。")
    st.stop()

# ─── パラメータ設定 ─────────────────────────────────────
temperature = st.slider(
    "Temperature",
    0.0, 2.0,
    value=DEFAULT_TEMPERATURE,
    step=0.1,
    disabled=locked
)
ntop_p = st.slider(
    "Top-p (Nucleus)",
    0.0, 1.0,
    value=DEFAULT_TOP_P,
    step=0.01,
    disabled=locked or temperature <= 0.0
)
ntop_k = st.slider(
    "Top-K Sampling",
    1, 50,
    value=DEFAULT_TOP_K,
    step=1,
    disabled=locked or ntop_p < 1.0 or temperature <= 0.0
)

# 状態メッセージ
st.markdown("---")
if locked:
    st.markdown("🔒 パラメータがロックされています。初期化で解除できます。")
elif temperature <= 0.0:
    st.markdown("⚠️ Temperature=0 のため Greedy Decoding です。")
elif ntop_p < 1.0:
    st.markdown("⚠️ Top-p Mode: Top-K 無効")
else:
    st.markdown("⚠️ Top-K Mode: Top-p 無効")

chart_placeholder = st.empty()
attention_placeholder = st.empty()

# ─── トークン生成とロックコールバック ────────────────────────
def generate_and_lock():
    ss = state
    # 生成前の選択ヘッドを保持
    old_idx = ss.step_index
    old_key = f'head_select_{old_idx}'
    prev_sel = ss.get(old_key, 'Average')

    # トークン生成
    result = generate_step(
        ss.input_ids, model, tokenizer,
        temperature, ntop_p, ntop_k
    )
    ss.input_ids = result["input_ids"]
    ss.steps.append(result["step_data"])

    # 新ステップのインデックス更新
    new_idx = len(ss.steps) - 1
    ss.step_index = new_idx

    # 新ステップでも前のヘッド選択を維持
    new_key = f'head_select_{new_idx}'
    ss[new_key] = prev_sel

    # ロック設定
    if not explore_mode:
        ss.lock_params = True

# トークン生成ボタン
st.button(
    "▶️ トークン生成",
    on_click=generate_and_lock,
    disabled=False
)


# ─── ステップナビゲーション & 可視化 ───────────────────────
if state.steps:
    idx = state.step_index
    step = state.steps[idx]

    # Prev/Next Buttons
    c1, _, c3 = st.columns([1, 2, 1])
    c1.button("← 前へ", on_click=lambda: setattr(state, 'step_index', max(idx-1,0)), disabled=(idx==0), key='prev')
    c3.button("次へ →", on_click=lambda: setattr(state, 'step_index', min(idx+1,len(state.steps)-1)), disabled=(idx==len(state.steps)-1), key='next')
    st.markdown(f"**Step {idx+1}/{len(state.steps)}**")

    # Top-K Plot
    if temperature <= 0.0:
        title = "Top-1 (Greedy)"
        limit = 1
    elif ntop_p < 1.0:
        title = f"Top-p Dist (p={ntop_p:.2f})"
        limit = 10
    else:
        title = f"Top-K Dist (k={ntop_k})"
        limit = ntop_k
    fig = plot_topk(
        tokens=step["tokens"],
        values=step["values"],
        ids=step["ids"],
        chosen=step["chosen"],
        top_k=limit,
        temperature=temperature,
        title=title
    )
    chart_placeholder.pyplot(fig)

    # Attention Heatmap
    attn = step["attn"]
    if attn.ndim == 2:
        attn = attn[np.newaxis, ...]
    options = ["Average"] + [f"Head {i}" for i in range(attn.shape[0])]
    widget_key = f"head_select_{idx}"
    sel = st.selectbox("Attention Head", options, key=widget_key)
    if sel == "Average":
        mat = attn.mean(axis=0)
    else:
        head_idx = int(sel.split()[1])
        mat = attn[head_idx]
    heat_fig = plot_attention(mat, step["all_toks"], title=sel)
    attention_placeholder.pyplot(heat_fig, clear_figure=False)

# ─── 最終出力を表示 ───────────────────────────────────────
st.markdown("### 🧠 最終アウトプット")
st.write(tokenizer.decode(state.input_ids[0], skip_special_tokens=True))
