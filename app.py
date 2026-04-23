"""
╔══════════════════════════════════════════════════════════════╗
║  محمد محمود — Hassaniya Chatbot (Streamlit App)             ║
║  Deploy the fine-tuned AraGPT2 Mohamed Mahmoud model        ║
╚══════════════════════════════════════════════════════════════╝

Setup:
    pip install streamlit transformers peft torch sentencepiece

Run:
    streamlit run app.py
"""

import streamlit as st
import torch
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="محمد محمود — شات بوت حساني",
    page_icon="🐪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for RTL Arabic + premium design ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@400;600;700&display=swap');

/* Global */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

/* Chat messages */
.user-msg, .bot-msg {
    padding: 1rem 1.2rem;
    border-radius: 16px;
    margin: 0.5rem 0;
    font-family: 'Noto Kufi Arabic', sans-serif;
    font-size: 1.05rem;
    line-height: 1.8;
    direction: rtl;
    text-align: right;
}
.user-msg {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    margin-left: 15%;
    border-bottom-left-radius: 4px;
}
.bot-msg {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    color: #e8e8e8;
    border: 1px solid rgba(255,255,255,0.12);
    margin-right: 15%;
    border-bottom-right-radius: 4px;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
    font-family: 'Noto Kufi Arabic', sans-serif;
}
.main-header h1 {
    color: #f0e6d3;
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}
.main-header p {
    color: #b8a88a;
    font-size: 1rem;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'Noto Kufi Arabic', sans-serif;
    margin-top: 0.5rem;
}
.status-online {
    background: rgba(72, 199, 142, 0.15);
    color: #48c78e;
    border: 1px solid rgba(72, 199, 142, 0.3);
}
.status-offline {
    background: rgba(255, 99, 71, 0.15);
    color: #ff6347;
    border: 1px solid rgba(255, 99, 71, 0.3);
}

/* Sidebar */
.sidebar-info {
    font-family: 'Noto Kufi Arabic', sans-serif;
    direction: rtl;
    text-align: right;
    color: #ccc;
    font-size: 0.9rem;
    line-height: 1.8;
}

/* Input area styling */
.stTextInput > div > div > input {
    font-family: 'Noto Kufi Arabic', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
    font-size: 1rem !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────────────
# We check multiple possible locations for the model folder
POSSIBLE_PATHS = [
    r"c:\Users\pc\Desktop\modele_ai\mohamed_mahmoud_model\final",
    os.path.join(os.path.dirname(__file__), "mohamed_mahmoud_model", "final"),
    "./mohamed_mahmoud_model/final"
]

MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

SYSTEM_PROMPT = (
    "أنت محمد محمود سيدي المختار، راعي إبل موريتاني من البادية. "
    "تتكلم الحسانية وتعرف كل شيء عن الإبل والصحراء والرعي. "
    "ما تعرف أمور المدينة والتكنولوجيا."
)


@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    if not MODEL_PATH:
        return None, None, f"❌ Model not found in any of the expected locations.\n\nPlease ensure the 'mohamed_mahmoud_model/final' folder exists."

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Load base model + LoRA adapter
        base_model_name = "aubmindlab/aragpt2-base"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()

        return model, tokenizer, None
    except Exception as e:
        return None, None, f"❌ Error loading model: {str(e)}"


def generate_response(model, tokenizer, question, max_new_tokens=80):
    """Generate a single, concise response from Mohamed Mahmoud."""
    prompt = (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>{question}<|end|>"
        f"<|assistant|>"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6, # Lowered slightly for more focused output
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.5, # Increased to prevent looping
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=end_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract response
    if "<|assistant|>" in generated:
        response = generated.split("<|assistant|>")[-1]
        if "<|end|>" in response:
            response = response.split("<|end|>")[0]
        
        # Comprehensive cleaning
        for tag in ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"]:
            response = response.replace(tag, "")
        response = response.strip()
    else:
        response = generated[len(prompt):].strip()

    # Ensure only ONE concise response block (stop at first repetition or new turn marker)
    # If the model starts repeating the user's question or generating a new turn, cut it
    if "👤" in response or "🐪" in response:
        response = response.split("👤")[0].split("🐪")[0].strip()

    # If the response is too long or contains multiple lines, take only the first relevant part
    lines = response.split("\n")
    if lines:
        response = lines[0].strip()

    return response or "ما فهمت قصدك، عاودلي السؤال."


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🐪 محمد محمود</h1>
    <p>راعي إبل موريتاني من البادية — يتكلم الحسانية</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
model, tokenizer, error = load_model()

if error:
    st.markdown(f"""
    <div class="main-header">
        <span class="status-badge status-offline">غير متصل</span>
    </div>
    """, unsafe_allow_html=True)
    st.error(error)
    st.info("""
    **📋 How to set up the model:**
    1. Train the model using `colab_train.py` on Google Colab
    2. Download the `mohamed_mahmoud_model.zip` file
    3. Unzip it into this project folder so you have:
       ```
       modele_ai/
       ├── app.py
       └── mohamed_mahmoud_model/
           └── final/
               ├── config.json
               ├── adapter_model.safetensors
               ├── tokenizer.json
               └── ...
       ```
    4. Run `streamlit run app.py`
    """)
    st.stop()

st.markdown("""
<div class="main-header">
    <span class="status-badge status-online">✦ متصل</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐪 عن محمد محمود")
    st.markdown("""
    <div class="sidebar-info">
    <strong>الاسم:</strong> محمد محمود سيدي المختار<br>
    <strong>المهنة:</strong> راعي إبل<br>
    <strong>المنطقة:</strong> البادية الموريتانية<br>
    <strong>اللهجة:</strong> الحسانية<br><br>
    يعرف كل شيء عن الإبل والصحراء والرعي.
    ما يعرف أمور المدينة والتكنولوجيا.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚙️ إعدادات")
    max_tokens = st.slider("طول الرد", 50, 300, 150, 25)
    temperature = st.slider("الإبداع", 0.1, 1.5, 0.7, 0.1)

    st.divider()

    st.markdown("### 💡 أمثلة")
    example_questions = [
        "السلام عليكم",
        "كيف الإبل عندك؟",
        "تعرف الإنترنت؟",
        "عندك حكمة من البادية؟",
        "كيف الجو اليوم؟",
        "أنت منه؟",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{q}"):
            st.session_state.example_input = q

# ── Chat history ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
    prefix = "👤" if msg["role"] == "user" else "🐪"
    st.markdown(
        f'<div class="{css_class}">{prefix} {msg["content"]}</div>',
        unsafe_allow_html=True
    )

# ── Input ────────────────────────────────────────────────────
# Check for example input from sidebar
input_value = ""
if "example_input" in st.session_state:
    input_value = st.session_state.example_input
    del st.session_state.example_input

user_input = st.text_input(
    "اكتب رسالتك هنا...",
    value=input_value,
    key="chat_input",
    placeholder="السلام عليكم يا محمد...",
    label_visibility="collapsed",
)

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(
        f'<div class="user-msg">👤 {user_input}</div>',
        unsafe_allow_html=True
    )

    # Generate response
    with st.spinner("محمد محمود يفكر... 🐪"):
        response = generate_response(model, tokenizer, user_input, max_new_tokens=max_tokens)

    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(
        f'<div class="bot-msg">🐪 {response}</div>',
        unsafe_allow_html=True
    )

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()
with col2:
    st.caption(f"💬 {len(st.session_state.messages)} رسالة")
