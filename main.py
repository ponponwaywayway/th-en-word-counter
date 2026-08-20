import streamlit as st
import pandas as pd
import operator
import altair as alt
from pythainlp import word_tokenize
from pythainlp.tag import pos_tag
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus.tnc import word_freqs as tnc_word_freqs

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Thai Word Counter & Frequency Analyzer",
    page_icon="📝",
    layout="wide"
)

# --- CSS จัดการกรอบสี่เหลี่ยมใหญ่แบบมีระยะขอบ + การ์ดสีขาว ---
st.markdown("""
<style>
    /* 1. พื้นหลัง Gradient พาสเทลทั้งหน้าจอ */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: linear-gradient(135deg, #d8e2fd 0%, #e2d9f3 35%, #eddcf4 70%, #fcdbe8 100%) !important;
        background-attachment: fixed !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* 2. กรอบสี่เหลี่ยมใหญ่: ลดความกว้าง + จัดกึ่งกลาง เว้นระยะขอบจอสวยงาม */
    .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 1200px !important;
        width: 90% !important;
        margin: 36px auto 48px auto !important;
        background: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(16px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
        border-radius: 36px !important;
        padding: 36px 40px !important;
        border: 1.5px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 0 16px 40px rgba(135, 120, 175, 0.12) !important;
    }

    /* 3. ล้างพื้นหลังส่วน Layout อื่นๆ */
    header, footer, [data-testid="stAppViewBlockContainer"], [data-testid="stHorizontalBlock"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* 4. สไตล์การ์ดสีขาวนูน (Solid White Cards) */
    .white-card {
        background-color: #ffffff;
        border-radius: 22px;
        padding: 20px 24px;
        box-shadow: 0 8px 24px rgba(135, 120, 175, 0.10);
        margin-bottom: 14px;
    }

    /* 5. สไตล์ช่องกรอกข้อมูลและ Selectbox */
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

    /* 6. ปุ่มประมวลผล */
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

    /* 7. Typography */
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
        font-size: 0.98rem;
        font-weight: 600;
        color: #484a63;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #232536;
        line-height: 1.1;
    }

    /* 8. สไตล์ตาราง Dataframe */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #edf0f7 !important;
        border-radius: 12px !important;
    }

    /* กล่อง Widget ฝังในสีขาว */
    .st-key-input_box, .st-key-table_box_1, .st-key-chart_box_1, 
    .st-key-table_box_2, .st-key-chart_box_2, .st-key-chart_box_pos {
        background: #ffffff !important;
        border-radius: 22px !important;
        padding: 24px !important;
        box-shadow: 0 8px 24px rgba(135, 120, 175, 0.10) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- คลัง Common / Stop Words ทั้งไทยและอังกฤษ ---
thai_stop = set(thai_stopwords())
english_stop = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'refer', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
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

# --- ดึงดัชนีความถี่คำมาตรฐานจากคลัง TNC ---
@st.cache_data
def get_tnc_corpus_freq():
    try:
        freqs = tnc_word_freqs()
        return dict(freqs) if freqs else {}
    except Exception:
        return {}

TNC_FREQ_DICT = get_tnc_corpus_freq()

def get_corpus_frequency(word: str) -> int:
    score = TNC_FREQ_DICT.get(word, 0)
    if word in ALL_COMMON_WORDS:
        score += 100_000_000
    return score

# --- ฟังก์ชันตัดและนับคำ + POS Tagging ---
def word_count(lyrics: str):
    if not lyrics.strip():
        return {}, {}, 0, {}, {}
    
    lyrics_token = word_tokenize(lyrics, keep_whitespace=False)
    sym = {'"', '[', ']', '(', ')', ',', '!', '.', '\n', '\s', ' ', '', 'ๆ', '?', ':', "'", '“', '”', '%', '-', '–', '—', '\\', '/', '>', '<', ';', '+', '*', '&'}
    lyrics_token_clean = []
    
    for w in lyrics_token:
        clean_str = ""
        for s in w:
            if s not in sym and not s.isalpha() and not s.isdigit():
                clean_str += s
            elif s.isalpha():
                clean_str += s.lower()
                
        if clean_str and clean_str not in sym and not clean_str.isdigit():
            lyrics_token_clean.append(clean_str)

    wordcount_all = {}
    wordcount_content = {}
    non_common_total_count = 0
    
    for w in lyrics_token_clean:
        wordcount_all[w] = wordcount_all.get(w, 0) + 1
        if w not in ALL_COMMON_WORDS:
            wordcount_content[w] = wordcount_content.get(w, 0) + 1
            non_common_total_count += 1

    sorted_all_list = sorted(
        wordcount_all.items(),
        key=lambda item: (-item[1], get_corpus_frequency(item[0]))
    )
    sorted_content_list = sorted(
        wordcount_content.items(),
        key=lambda item: (-item[1], get_corpus_frequency(item[0]))
    )

    # POS Tagging (orchid_ud)
    list_of_words = [k for k, v in sorted_all_list]
    postag = pos_tag(list_of_words, corpus="orchid_ud") if list_of_words else []
    
    word_to_pos = {w: tag for w, tag in postag}
    
    pos_dict = {}
    for w, tag in postag:
        pos_dict[tag] = pos_dict.get(tag, 0) + 1

    pos_dict_sorted = dict(sorted(pos_dict.items(), key=operator.itemgetter(1), reverse=True))

    return dict(sorted_all_list), dict(sorted_content_list), non_common_total_count, word_to_pos, pos_dict_sorted

# --- จัดการ Session State ---
if "wc_all" not in st.session_state:
    st.session_state.wc_all = None

if "wc_content" not in st.session_state:
    st.session_state.wc_content = None

if "non_common_total" not in st.session_state:
    st.session_state.non_common_total = 0

if "word_to_pos" not in st.session_state:
    st.session_state.word_to_pos = {}

if "pos_dict_sorted" not in st.session_state:
    st.session_state.pos_dict_sorted = None

if "history_list" not in st.session_state:
    st.session_state.history_list = []

if "current_text" not in st.session_state:
    st.session_state.current_text = ""

def apply_history():
    selected = st.session_state.selected_history
    if selected and selected != "-- เลือกดูประวัติข้อความเก่า --":
        st.session_state.current_text = selected
        all_w, content_w, nc, w_pos, p_dict = word_count(selected)
        st.session_state.wc_all = all_w
        st.session_state.wc_content = content_w
        st.session_state.non_common_total = nc
        st.session_state.word_to_pos = w_pos
        st.session_state.pos_dict_sorted = p_dict

# ==================== แถวที่ 1 (ซ้าย: Input Card, ขวา: 3 Metric Cards) ====================
r1_left, r1_right = st.columns([1.3, 1], gap="medium")

with r1_left:
    with st.container(key="input_box"):
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
                
                all_w, content_w, nc, w_pos, p_dict = word_count(text_input)
                st.session_state.wc_all = all_w
                st.session_state.wc_content = content_w
                st.session_state.non_common_total = nc
                st.session_state.word_to_pos = w_pos
                st.session_state.pos_dict_sorted = p_dict
                st.rerun()
            else:
                st.session_state.wc_all = None
                st.session_state.wc_content = None
                st.session_state.non_common_total = 0
                st.session_state.word_to_pos = {}
                st.session_state.pos_dict_sorted = None

with r1_right:
    total_tokens = sum(st.session_state.wc_all.values()) if st.session_state.wc_all else 0
    unique_tokens = len(st.session_state.wc_all) if st.session_state.wc_all else 0
    non_common_words = st.session_state.non_common_total if st.session_state.wc_all else 0
    
    st.markdown(f"""
    <div class="white-card">
        <div class="metric-title">จำนวนคำทั้งหมด (Total Tokens)</div>
        <div class="metric-value">{total_tokens:,}</div>
    </div>
    <div class="white-card">
        <div class="metric-title">จำนวนคำที่ไม่ซ้ำกัน (Unique Words)</div>
        <div class="metric-value">{unique_tokens:,}</div>
    </div>
    <div class="white-card">
        <div class="metric-title">คำเฉพาะ / ไม่ใช่คำทั่วไป (Non-Common Words)</div>
        <div class="metric-value">{non_common_words:,}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 2 (ชุดที่ 1: คำทั้งหมด All Words) ====================
r2_left, r2_right = st.columns([1, 1.3], gap="medium")

if st.session_state.wc_all:
    data_all = []
    for idx, (word, count) in enumerate(st.session_state.wc_all.items(), start=1):
        pos = st.session_state.word_to_pos.get(word, "-")
        data_all.append({"ลำดับ": idx, "WORD": word, "POS TAG": pos, "COUNT": count})
    df_all = pd.DataFrame(data_all)
else:
    df_all = pd.DataFrame(columns=["ลำดับ", "WORD", "POS TAG", "COUNT"])

with r2_left:
    with st.container(key="table_box_1"):
        st.markdown('<div class="card-title">📊 ตารางแจกแจงความถี่ (คำทั้งหมด)</div>', unsafe_allow_html=True)
        st.dataframe(df_all, hide_index=True, use_container_width=True, height=280)

with r2_right:
    with st.container(key="chart_box_1"):
        st.markdown('<div class="card-title">📈 คำที่พบมากที่สุด (Top 15 - รวมทุกคำ)</div>', unsafe_allow_html=True)
        if not df_all.empty:
            top_15_all = df_all.head(15)
            
            bars = alt.Chart(top_15_all).mark_bar(
                color="#f59e0b",
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
                width=14
            ).encode(
                x=alt.X("WORD:N", sort=top_15_all["WORD"].tolist(), axis=alt.Axis(labelAngle=90, labelColor="#475569", title=None, tickColor="#cbd5e1")),
                y=alt.Y("COUNT:Q", axis=alt.Axis(labelColor="#475569", title=None, gridColor="#f1f5f9", tickColor="#cbd5e1"))
            )
            
            text_labels = alt.Chart(top_15_all).mark_text(
                align='center',
                baseline='bottom',
                dy=-4,
                color='#475569',
                fontSize=11,
                fontWeight=600
            ).encode(
                x=alt.X("WORD:N", sort=top_15_all["WORD"].tolist()),
                y=alt.Y("COUNT:Q"),
                text=alt.Text("COUNT:Q")
            )
            
            chart_all = (bars + text_labels).properties(height=260, background="#ffffff").configure_view(strokeWidth=0)
            st.altair_chart(chart_all, use_container_width=True)
        else:
            st.markdown("<p style='color: #8a8ca3; height: 260px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 3 (ชุดที่ 2: เฉพาะคำที่ไม่ใช่ Stop Words) ====================
r3_left, r3_right = st.columns([1, 1.3], gap="medium")

if st.session_state.wc_content:
    data_content = []
    for idx, (word, count) in enumerate(st.session_state.wc_content.items(), start=1):
        pos = st.session_state.word_to_pos.get(word, "-")
        data_content.append({"ลำดับ": idx, "WORD": word, "POS TAG": pos, "COUNT": count})
    df_content = pd.DataFrame(data_content)
else:
    df_content = pd.DataFrame(columns=["ลำดับ", "WORD", "POS TAG", "COUNT"])

with r3_left:
    with st.container(key="table_box_2"):
        st.markdown('<div class="card-title">🔍 ตารางแจกแจงความถี่ (ไม่รวม Stop Words)</div>', unsafe_allow_html=True)
        st.dataframe(df_content, hide_index=True, use_container_width=True, height=280)

with r3_right:
    with st.container(key="chart_box_2"):
        st.markdown('<div class="card-title">✨ คำสำคัญที่พบมากที่สุด (Top 15 - ไม่รวม Stop Words)</div>', unsafe_allow_html=True)
        if not df_content.empty:
            top_15_content = df_content.head(15)
            
            bars_content = alt.Chart(top_15_content).mark_bar(
                color="#6366f1",
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
                width=14
            ).encode(
                x=alt.X("WORD:N", sort=top_15_content["WORD"].tolist(), axis=alt.Axis(labelAngle=90, labelColor="#475569", title=None, tickColor="#cbd5e1")),
                y=alt.Y("COUNT:Q", axis=alt.Axis(labelColor="#475569", title=None, gridColor="#f1f5f9", tickColor="#cbd5e1"))
            )
            
            text_labels_content = alt.Chart(top_15_content).mark_text(
                align='center',
                baseline='bottom',
                dy=-4,
                color='#475569',
                fontSize=11,
                fontWeight=600
            ).encode(
                x=alt.X("WORD:N", sort=top_15_content["WORD"].tolist()),
                y=alt.Y("COUNT:Q"),
                text=alt.Text("COUNT:Q")
            )
            
            chart_content = (bars_content + text_labels_content).properties(height=260, background="#ffffff").configure_view(strokeWidth=0)
            st.altair_chart(chart_content, use_container_width=True)
        else:
            st.markdown("<p style='color: #8a8ca3; height: 260px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 4 (ชุดที่ 3: กราฟ POS Tag Frequency แบบเต็มแถว) ====================
if st.session_state.pos_dict_sorted:
    df_pos_summary = pd.DataFrame(list(st.session_state.pos_dict_sorted.items()), columns=["POS TAG", "COUNT"])
else:
    df_pos_summary = pd.DataFrame(columns=["POS TAG", "COUNT"])

with st.container(key="chart_box_pos"):
    st.markdown('<div class="card-title">📊 สัดส่วนชนิดของคำที่พบ (POS Tag Frequency)</div>', unsafe_allow_html=True)
    if not df_pos_summary.empty:
        bars_pos = alt.Chart(df_pos_summary).mark_bar(
            color="#ec4899",
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
            width=22
        ).encode(
            x=alt.X("POS TAG:N", sort=df_pos_summary["POS TAG"].tolist(), axis=alt.Axis(labelAngle=0, labelColor="#475569", title=None, tickColor="#cbd5e1")),
            y=alt.Y("COUNT:Q", axis=alt.Axis(labelColor="#475569", title=None, gridColor="#f1f5f9", tickColor="#cbd5e1"))
        )
        
        text_labels_pos = alt.Chart(df_pos_summary).mark_text(
            align='center',
            baseline='bottom',
            dy=-4,
            color='#475569',
            fontSize=11,
            fontWeight=600
        ).encode(
            x=alt.X("POS TAG:N", sort=df_pos_summary["POS TAG"].tolist()),
            y=alt.Y("COUNT:Q"),
            text=alt.Text("COUNT:Q")
        )
        
        chart_pos = (bars_pos + text_labels_pos).properties(height=260, background="#ffffff").configure_view(strokeWidth=0)
        st.altair_chart(chart_pos, use_container_width=True)
    else:
        st.markdown("<p style='color: #8a8ca3; height: 160px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)
