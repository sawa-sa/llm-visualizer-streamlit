import numpy as np
import streamlit as st
from config import DEFAULT_PROMPTS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K
from model_loader import load_model
from generator import generate_step
from visualizer import plot_topk, plot_attention

# ─── セッション状態の初期化 ───────────────────────────────
if "input_ids" not in st.session_state:
    st.session_state.input_ids = None
if "generated_tokens" not in st.session_state:
    st.session_state.generated_tokens = []
if "steps" not in st.session_state:
    st.session_state.steps = []
if "step_index" not in st.session_state:
    st.session_state.step_index = 0
if "prompt" not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPTS[0]
if "prompt_initialized" not in st.session_state:
    st.session_state.prompt_initialized = False
if "head_select" not in st.session_state:
    st.session_state.head_select = "Average"
if "lock_params" not in st.session_state:
    st.session_state.lock_params = False

# ─── モデルロード ─────────────────────────────────────────
model, tokenizer = load_model()

# ─── UI: 探索モードとロック判定 ─────────────────────────────
explore_mode = st.checkbox(
    "🔀 探索モード: 途中でパラメータ変更を許可",
    value=False,
    help="オフにすると生成開始後に全パラメータをロックします"
)
generation_started = st.session_state.lock_params and not explore_mode

st.title("🔍 GPT-2 Medium 可視化デモ")

# ─── プロンプト初期化のコールバック ─────────────────────────
def init_prompt():
    ss = st.session_state
    ss.input_ids = tokenizer.encode(ss.prompt, return_tensors="pt")
    ss.generated_tokens = []
    ss.steps = []
    ss.step_index = 0
    ss.prompt_initialized = False
    ss.head_select = "Average"
    ss.lock_params = False

# プロンプト選択＆入力
example_prompt = st.selectbox(
    "🧪 試してみたいプロンプトを選んでください（編集も可能）",
    ["（←選んでください）"] + DEFAULT_PROMPTS,
    key="prompt_selector",
    disabled=generation_started
)
if example_prompt != "（←選んでください）" and not st.session_state.prompt_initialized:
    st.session_state.prompt = example_prompt
    st.session_state.prompt_initialized = True

prompt = st.text_input(
    "プロンプト",
    value=st.session_state.prompt,
    disabled=generation_started
)

# プロンプト初期化ボタン
st.button(
    "🔄 プロンプト初期化",
    on_click=init_prompt,
    disabled=False
)

# 初期生成前のチェック
if st.session_state.input_ids is None:
    st.warning("まずはプロンプトを初期化してください。")
    st.stop()

# ─── パラメータスライダー ─────────────────────────────────
temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=DEFAULT_TEMPERATURE,
    step=0.1,
    disabled=generation_started
)
ntop_p = st.slider(
    "Top-p (Nucleus sampling)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_TOP_P,
    step=0.01,
    help="Top-p < 1.0 のときは Top-p サンプリング、Top-p = 1.0 の時は Top-K サンプリング",
    disabled=generation_started or temperature < 1e-5
)
ntop_k = st.slider(
    "Top-K sampling",
    min_value=1,
    max_value=50,
    value=DEFAULT_TOP_K,
    step=1,
    help="Top-p = 1.0 のときのみ有効",
    disabled=generation_started or ntop_p < 1.0 or temperature < 1e-5
)

st.markdown("---")
if generation_started:
    st.markdown("🔒 生成開始後はパラメータ変更不可です。リセットで解除。")
elif temperature < 1e-5:
    st.markdown("⚠️ Temperature=0 のため Top-p と Top-K は無効です")
elif ntop_p < 1.0:
    st.markdown("⚠️ Top-K は現在無効です（Top-p 有効）")
else:
    st.markdown("⚠️ Top-p は現在無効です（Top-K 有効）")

chart_placeholder = st.empty()
attention_placeholder = st.empty()

# ─── トークン生成のコールバック ─────────────────────────────
def do_generate(temp, top_p, top_k):
    ss = st.session_state
    result = generate_step(
        ss.input_ids,
        model,
        tokenizer,
        temp,
        top_p,
        top_k
    )
    ss.input_ids = result["input_ids"]
    ss.steps.append(result["step_data"])
    ss.step_index = len(ss.steps) - 1
    if not explore_mode:
        ss.lock_params = True

# ─── トークン生成のコールバック ─────────────────────────────

def do_generate(temp, top_p, top_k):
    ss = st.session_state
    result = generate_step(
        ss.input_ids,
        model,
        tokenizer,
        temp,
        top_p,
        top_k
    )
    ss.input_ids = result["input_ids"]
    ss.steps.append(result["step_data"])
    ss.step_index = len(ss.steps) - 1
    if not explore_mode:
        ss.lock_params = True

# トークン生成ボタン
st.button(
    "▶️ トークン生成",
    on_click=do_generate,
    args=(temperature, ntop_p, ntop_k),
    disabled=False
)

# ─── ステップナビゲーションと可視化と可視化 ─────────────────────────
if st.session_state.steps:
    ss = st.session_state
    idx = ss.step_index
    step = ss.steps[idx]

    c1, _, c3 = st.columns([1, 2, 1])
    c1.button(
        "← 前へ",
        on_click=lambda: ss.update(step_index=max(ss.step_index-1,0)),
        disabled=(idx==0),
        key="prev"
    )
    c3.button(
        "次へ →",
        on_click=lambda: ss.update(step_index=min(ss.step_index+1,len(ss.steps)-1)),
        disabled=(idx==len(ss.steps)-1),
        key="next"
    )
    st.markdown(f"**Step {idx+1}/{len(ss.steps)}**")

    # Top-K グラフ
    if temperature < 1e-5:
        title = "Top-1 (Greedy decoding)"
        limit = 1
    elif ntop_p < 1.0:
        title = f"Top-p Distribution (p={ntop_p:.2f})"
        limit = 10
    else:
        title = f"Top-K Distribution (k={ntop_k})"
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

    # Attention ヒートマップ
    attn = step["attn"]
    if attn.ndim == 2:
        attn = attn[np.newaxis,...]
    options = ["Average"] + [f"Head {i}" for i in range(attn.shape[0])]
    sel = st.selectbox(
        "Attention Head を選択",
        options,
        key="head_select",
        index=options.index(ss.head_select)
    )
    if sel != ss.head_select:
        ss.head_select = sel
    mat = attn.mean(axis=0) if sel == "Average" else attn[int(sel.split()[1])]
    heat = plot_attention(mat, step["all_toks"], title=sel)
    attention_placeholder.pyplot(heat, clear_figure=False)

# ─── 最終出力文表示 ───────────────────────────────────────
st.markdown("### 🧠 最終的な出力文")
st.write(tokenizer.decode(st.session_state.input_ids[0], skip_special_tokens=True))
