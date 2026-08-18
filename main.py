import streamlit as st
import pandas as pd
import operator
import altair as alt
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Thai Word Counter & Frequency Analyzer",
    page_icon="📝",
    layout="wide"
)

# --- CSS จัดการพื้นหลัง Gradient และการ์ดสีขาวทึบ ไร้เส้นขอบ ---
st.markdown("""
<style>
    /* 1. พื้นหลัง Gradient พาสเทลทั้งหน้าจอ */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: linear-gradient(135deg, #d8e2fd 0%, #e2d9f3 35%, #eddcf4 70%, #fcdbe8 100%) !important;
        background-attachment: fixed !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* 2. ล้างกรอบและพื้นหลังระดับ Page Layout */
    header, footer, .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stHorizontalBlock"],
    [data-testid="column"],
    [data-testid="stColumn"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* 3. การ์ดสีขาวทึบ ไร้ขอบเทาอย่างเด็ดขาด */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border: 0px solid transparent !important;
        border-style: none !important;
        outline: none !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 24px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px rgba(135, 120, 175, 0.16) !important;
    }

    /* 4. ช่อง Text Area & Selectbox */
    .stTextArea textarea, div[data-baseweb="select"] > div {
        background: #fbfbfe !important;
        border: 1.5px solid #e2e5f0 !important;
        border-radius: 14px !important;
        color: #2b2d42 !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea {
        padding: 12px !important;
    }
    .stTextArea textarea:focus {
        border-color: #7b7393 !important;
        box-shadow: 0 0 0 2px rgba(123, 115, 147, 0.15) !important;
    }
    .stTextArea label p, .stSelectbox label p {
        color: #555770 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* 5. ปุ่มประมวลผลสีกรมท่าเข้ม */
    div.stButton > button {
        background: #34324b !important;
        color: #ffffff !important;
        border-radius: 20px !important;
        padding: 6px 38px !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        border: none !important;
        display: block !important;
        margin: 14px auto 0 auto !important;
        box-shadow: 0 4px 12px rgba(52, 50, 75, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: #232136 !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    /* 6. Typography */
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #2b2d42;
        margin-bottom: 2px;
    }
    .card-subtitle {
        font-size: 0.85rem;
        color: #7b7d96;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 1.02rem;
        font-weight: 600;
        color: #484a63;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #232536;
        line-height: 1.1;
    }

    /* 7. Dataframe Light Theme */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #edf0f7 !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- คลัง Common / Stop Words ทั้งไทยและอังกฤษ ---
thai_stop = set(thai_stopwords())
english_stop = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
    "wouldn't", 'oh', 'baby', 'yeah', 'la', 'na', 'ah', 'ooh', 'whoa', 'hey', 'uh', 'um'
}
ALL_COMMON_WORDS = thai_stop.union(english_stop)

# --- ฟังก์ชันตัดและนับคำ ---
def word_count(lyrics: str):
    if not lyrics.strip():
        return {}, {}, 0
    
    lyrics_token = word_tokenize(lyrics, keep_whitespace=False)
    sym = {'"', '[', ']', '(', ')', ',', '!', '.', '\n', '\s', ' ', '', 'ๆ', '?', ':', "'"}
    lyrics_token_clean = []
    
    for w in lyrics_token:
        clean_str = ""
        for s in w:
            if s not in sym and not s.isalpha():
                clean_str += s
            elif s.isalpha():
                clean_str += s.lower()
        if clean_str and clean_str not in sym:
            lyrics_token_clean.append(clean_str)

    wordcount_all = {}
    wordcount_content = {}
    non_common_total_count = 0
    
    for w in lyrics_token_clean:
        wordcount_all[w] = wordcount_all.get(w, 0) + 1
        if w not in ALL_COMMON_WORDS:
            wordcount_content[w] = wordcount_content.get(w, 0) + 1
            non_common_total_count += 1

    sorted_all = dict(sorted(wordcount_all.items(), key=operator.itemgetter(1), reverse=True))
    sorted_content = dict(sorted(wordcount_content.items(), key=operator.itemgetter(1), reverse=True))
    
    return sorted_all, sorted_content, non_common_total_count

# --- จัดการ Session State ---
if "wc_all" not in st.session_state:
    st.session_state.wc_all = None

if "wc_content" not in st.session_state:
    st.session_state.wc_content = None

if "non_common_total" not in st.session_state:
    st.session_state.non_common_total = 0

if "history_list" not in st.session_state:
    st.session_state.history_list = []

if "current_text" not in st.session_state:
    st.session_state.current_text = ""

def apply_history():
    selected = st.session_state.selected_history
    if selected and selected != "-- เลือกดูประวัติข้อความเก่า --":
        st.session_state.current_text = selected
        all_w, content_w, nc = word_count(selected)
        st.session_state.wc_all = all_w
        st.session_state.wc_content = content_w
        st.session_state.non_common_total = nc

# ==================== แถวที่ 1 (ซ้าย: Input Card, ขวา: 3 Metric Cards) ====================
r1_left, r1_right = st.columns([1.3, 1], gap="medium")

with r1_left:
    with st.container(border=True):
        st.markdown('<div class="card-title">📝 Thai Word Counter & Frequency Analyzer</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">วางเนื้อเพลงหรือข้อความภาษาไทยหรืออังกฤษเพื่อวิเคราะห์และนับความถี่ของคำ</div>', unsafe_allow_html=True)
        
        if st.session_state.history_list:
            history_options = ["-- เลือกดูประวัติข้อความเก่า --"] + st.session_state.history_list
            st.selectbox(
                label="📜 ประวัติข้อความที่เคยใส่:",
                options=history_options,
                format_func=lambda x: (x[:45] + "...") if len(x) > 45 else x,
                key="selected_history",
                on_change=apply_history
            )
        
        text_input = st.text_area(
            label="กรอกหรือวางข้อความที่ต้องการนับคำที่นี่:",
            value=st.session_state.current_text,
            placeholder="วางเนื้อหาหรือข้อความยาว ๆ ลงในช่องนี้...",
            height=180
        )
        
        if st.button("ประมวลผล"):
            st.session_state.current_text = text_input
            if text_input.strip():
                if text_input in st.session_state.history_list:
                    st.session_state.history_list.remove(text_input)
                st.session_state.history_list.insert(0, text_input)
                
                all_w, content_w, nc = word_count(text_input)
                st.session_state.wc_all = all_w
                st.session_state.wc_content = content_w
                st.session_state.non_common_total = nc
                st.rerun()
            else:
                st.session_state.wc_all = None
                st.session_state.wc_content = None
                st.session_state.non_common_total = 0

with r1_right:
    total_tokens = sum(st.session_state.wc_all.values()) if st.session_state.wc_all else 0
    unique_tokens = len(st.session_state.wc_all) if st.session_state.wc_all else 0
    non_common_words = st.session_state.non_common_total if st.session_state.wc_all else 0
    
    with st.container(border=True):
        st.markdown(f"""
        <div class="metric-title">จำนวนคำทั้งหมด (Total Tokens)</div>
        <div class="metric-value">{total_tokens:,}</div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown(f"""
        <div class="metric-title">จำนวนคำที่ไม่ซ้ำกัน (Unique Words)</div>
        <div class="metric-value">{unique_tokens:,}</div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"""
        <div class="metric-title">คำเฉพาะ / ไม่ใช่คำทั่วไป (Non-Common Words)</div>
        <div class="metric-value">{non_common_words:,}</div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 2 (ชุดที่ 1: คำทั้งหมด All Words) ====================
r2_left, r2_right = st.columns([1, 1.3], gap="medium")

if st.session_state.wc_all:
    df_all = pd.DataFrame(list(st.session_state.wc_all.items()), columns=["คำ (WORD)", "จำนวนครั้งที่พบ"])
    df_all.insert(0, "ลำดับ", range(1, len(df_all) + 1))
else:
    df_all = pd.DataFrame(columns=["ลำดับ", "คำ (WORD)", "จำนวนครั้งที่พบ"])

with r2_left:
    with st.container(border=True):
        st.markdown('<div class="card-title">📊 ตารางแจกแจงความถี่ (คำทั้งหมด)</div>', unsafe_allow_html=True)
        st.dataframe(df_all, hide_index=True, use_container_width=True, height=280)

with r2_right:
    with st.container(border=True):
        st.markdown('<div class="card-title">📈 คำที่พบมากที่สุด (Top 15 - รวมทุกคำ)</div>', unsafe_allow_html=True)
        if not df_all.empty:
            top_15_all = df_all.head(15)
            
            bars = alt.Chart(top_15_all).mark_bar(
                color="#f59e0b",
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
                width=14
            ).encode(
                x=alt.X("คำ (WORD):N", sort=None, axis=alt.Axis(labelAngle=90, labelColor="#475569", title=None, tickColor="#cbd5e1")),
                y=alt.Y("จำนวนครั้งที่พบ:Q", axis=alt.Axis(labelColor="#475569", title=None, gridColor="#f1f5f9", tickColor="#cbd5e1"))
            )
            
            text_labels = alt.Chart(top_15_all).mark_text(
                align='center',
                baseline='bottom',
                dy=-4,
                color='#475569',
                fontSize=11,
                fontWeight=600
            ).encode(
                x=alt.X("คำ (WORD):N", sort=None),
                y=alt.Y("จำนวนครั้งที่พบ:Q"),
                text=alt.Text("จำนวนครั้งที่พบ:Q")
            )
            
            chart_all = (bars + text_labels).properties(height=260, background="#ffffff").configure_view(strokeWidth=0)
            st.altair_chart(chart_all, use_container_width=True)
        else:
            st.markdown("<p style='color: #8a8ca3; height: 260px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 3 (ชุดที่ 2: เฉพาะคำที่ไม่ใช่ Stop Words) ====================
r3_left, r3_right = st.columns([1, 1.3], gap="medium")

if st.session_state.wc_content:
    df_content = pd.DataFrame(list(st.session_state.wc_content.items()), columns=["คำ (WORD)", "จำนวนครั้งที่พบ"])
    df_content.insert(0, "ลำดับ", range(1, len(df_content) + 1))
else:
    df_content = pd.DataFrame(columns=["ลำดับ", "คำ (WORD)", "จำนวนครั้งที่พบ"])

with r3_left:
    with st.container(border=True):
        st.markdown('<div class="card-title">🔍 ตารางแจกแจงความถี่ (ไม่รวม Stop Words)</div>', unsafe_allow_html=True)
        st.dataframe(df_content, hide_index=True, use_container_width=True, height=280)

with r3_right:
    with st.container(border=True):
        st.markdown('<div class="card-title">✨ คำสำคัญที่พบมากที่สุด (Top 15 - ไม่รวม Stop Words)</div>', unsafe_allow_html=True)
        if not df_content.empty:
            top_15_content = df_content.head(15)
            
            bars_content = alt.Chart(top_15_content).mark_bar(
                color="#6366f1",
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
                width=14
            ).encode(
                x=alt.X("คำ (WORD):N", sort=None, axis=alt.Axis(labelAngle=90, labelColor="#475569", title=None, tickColor="#cbd5e1")),
                y=alt.Y("จำนวนครั้งที่พบ:Q", axis=alt.Axis(labelColor="#475569", title=None, gridColor="#f1f5f9", tickColor="#cbd5e1"))
            )
            
            text_labels_content = alt.Chart(top_15_content).mark_text(
                align='center',
                baseline='bottom',
                dy=-4,
                color='#475569',
                fontSize=11,
                fontWeight=600
            ).encode(
                x=alt.X("คำ (WORD):N", sort=None),
                y=alt.Y("จำนวนครั้งที่พบ:Q"),
                text=alt.Text("จำนวนครั้งที่พบ:Q")
            )
            
            chart_content = (bars_content + text_labels_content).properties(height=260, background="#ffffff").configure_view(strokeWidth=0)
            st.altair_chart(chart_content, use_container_width=True)
        else:
            st.markdown("<p style='color: #8a8ca3; height: 260px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)
