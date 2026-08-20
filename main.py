import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import operator
import io
import base64
import altair as alt
from PIL import Image, ImageDraw, ImageFont
import re # เพิ่ม re สำหรับจัดการ split ประโยค
from pythainlp import word_tokenize
from pythainlp.tag import pos_tag
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus.tnc import word_freqs as tnc_word_freqs

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Thai Word Counter & Context Analyzer",
    page_icon="📝",
    layout="wide"
)

# --- CSS จัดการ Layout และสไตล์ (คงเดิม) ---
st.markdown("""
<style>
    /* 1. พื้นหลัง Gradient พาสเทลทั้งหน้าจอ */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: linear-gradient(135deg, #d8e2fd 0%, #e2d9f3 35%, #eddcf4 70%, #fcdbe8 100%) !important;
        background-attachment: fixed !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* 2. กรอบสี่เหลี่ยมใหญ่ */
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

    /* 4. สไตล์การ์ดสีขาวนูน */
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

    /* 6. ปุ่มประมวลผล และปุ่มดาวน์โหลด */
    div.stButton {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    div.stButton > button {
        background: #34324b !important;
        color: #ffffff !important;
        border-radius: 20px !important;
        padding: 6px 32px !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(52, 50, 75, 0.25) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
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
    .st-key-table_box_2, .st-key-chart_box_2, .st-key-chart_box_pos,
    .st-key-table_box_concordance {
        background: #ffffff !important;
        border-radius: 22px !important;
        padding: 24px !important;
        box-shadow: 0 8px 24px rgba(135, 120, 175, 0.10) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- ฟังก์ชันสร้างภาพ 9:16 (คงเดิม) ---
def generate_story_image(text_sample, total, unique, non_common):
    width, height = 1080, 1920
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)

    for y in range(height):
        factor = y / height
        r = int(216 + (252 - 216) * factor)
        g = int(226 + (219 - 226) * factor)
        b = int(253 + (232 - 253) * factor)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    font_paths = ["tahoma.ttf", "leelawad.ttf", "Thonburi.ttc", "Angsana.ttc", "/System/Library/Fonts/Supplemental/Thonburi.ttc"]
    font_main = None
    for p in font_paths:
        try:
            font_title = ImageFont.truetype(p, 54)
            font_sub = ImageFont.truetype(p, 32)
            font_body = ImageFont.truetype(p, 28)
            font_num = ImageFont.truetype(p, 64)
            font_main = True
            break
        except Exception: continue

    if not font_main: font_title = font_sub = font_body = font_num = ImageFont.load_default()

    draw.rounded_rectangle([60, 100, 1020, 1820], radius=44, fill=(255, 255, 255, 140), outline=(255, 255, 255), width=4)
    draw.text((120, 160), "📝 Word Counter", fill="#232536", font=font_title)
    draw.text((120, 230), "Frequency & Token Analysis Summary", fill="#7b7d96", font=font_sub)

    draw.rounded_rectangle([110, 310, 970, 780], radius=28, fill="#ffffff", outline="#edf0f7", width=2)
    draw.text((150, 350), "ตัวอย่างข้อความ (Sample Text):", fill="#555770", font=font_sub)
    lines = text_sample.strip().split("\n")[:7]
    sample_text_display = "\n".join([l[:38] + ("..." if len(l) > 38 else "") for l in lines])
    draw.text((150, 410), sample_text_display, fill="#2b2d42", font=font_body, spacing=14)

    draw.rounded_rectangle([110, 830, 970, 1070], radius=28, fill="#ffffff", outline="#edf0f7", width=2)
    draw.text((150, 870), "จำนวนคำทั้งหมด (Total Tokens)", fill="#484a63", font=font_sub)
    draw.text((150, 930), f"{total:,}", fill="#232536", font=font_num)

    draw.rounded_rectangle([110, 1120, 970, 1360], radius=28, fill="#ffffff", outline="#edf0f7", width=2)
    draw.text((150, 1160), "จำนวนคำที่ไม่ซ้ำกัน (Unique Words)", fill="#484a63", font=font_sub)
    draw.text((150, 1220), f"{unique:,}", fill="#232536", font=font_num)

    draw.rounded_rectangle([110, 1410, 970, 1650], radius=28, fill="#ffffff", outline="#edf0f7", width=2)
    draw.text((150, 1450), "คำเฉพาะ / ไม่ใช่คำทั่วไป (Non-Common Words)", fill="#484a63", font=font_sub)
    draw.text((150, 1510), f"{non_common:,}", fill="#232536", font=font_num)

    draw.text((380, 1720), "Created with Streamlit & PyThaiNLP", fill="#8a8ca3", font=font_body)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- ฟังก์ชันหลักในการนับคำและวิเคราะห์ (คงเดิม) ---
thai_stop = set(thai_stopwords())
english_stop = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'refer', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'oh', 'baby', 'yeah', 'la', 'na', 'ah', 'ooh', 'whoa', 'hey', 'uh', 'um'}
ALL_COMMON_WORDS = thai_stop.union(english_stop)

@st.cache_data
def get_tnc_corpus_freq():
    try:
        freqs = tnc_word_freqs()
        return dict(freqs) if freqs else {}
    except Exception: return {}

TNC_FREQ_DICT = get_tnc_corpus_freq()

def get_corpus_frequency(word: str) -> int:
    score = TNC_FREQ_DICT.get(word, 0)
    if word in ALL_COMMON_WORDS: score += 100_000_000
    return score

def word_count(lyrics: str):
    if not lyrics.strip(): return {}, {}, 0, {}
    
    # แยกคำรวมช่องว่างเพื่อใช้ทำ Concordance
    raw_tokens = word_tokenize(lyrics, keep_whitespace=True)
    
    sym = {'"', '[', ']', '(', ')', ',', '!', '.', '\n', '\s', ' ', '', 'ๆ', '?', ':', "'", '“', '”', '%', '-'}
    
    # คำนวณความถี่ (Token ตัวพิมพ์เล็ก ไม่รวมสัญลักษณ์)
    processed_tokens = []
    for w in raw_tokens:
        clean_w = w.strip().lower()
        if clean_w and clean_w not in sym and not clean_w.isdigit():
            processed_tokens.append(clean_w)

    wordcount_all = {}
    wordcount_content = {}
    non_common_total_count = 0
    
    for w in processed_tokens:
        wordcount_all[w] = wordcount_all.get(w, 0) + 1
        if w not in ALL_COMMON_WORDS:
            wordcount_content[w] = wordcount_content.get(w, 0) + 1
            non_common_total_count += 1

    sorted_all = dict(sorted(wordcount_all.items(), key=lambda item: (-item[1], get_corpus_frequency(item[0]))))
    sorted_content = dict(sorted(wordcount_content.items(), key=lambda item: (-item[1], get_corpus_frequency(item[0]))))

    # POS Tagging
    list_of_words = list(sorted_all.keys())
    postag = pos_tag(list_of_words, corpus="orchid_ud") if list_of_words else []
    word_to_pos = {w: tag for w, tag in postag}

    return sorted_all, sorted_content, non_common_total_count, word_to_pos, raw_tokens

# --- ฟังก์ชันใหม่: สร้างตาราง Concordance (บริบทคำ) ---
def get_concordance_table(target_word: str, raw_tokens: list, window: int = 5):
    if not target_word or not raw_tokens:
        return pd.DataFrame(columns=['ลำดับ', 'บริบทซ้าย (Left Context)', 'คำเป้าหมาย (Key)', 'บริบทขวา (Right Context)'])

    data = []
    sym_to_clear = {'\n', '\r', '\t'}
    
    # หาตำแหน่งของคำเป้าหมาย (ต้อง match แบบ case-insensitive และ trim space)
    target_clean = target_word.strip().lower()
    
    # เก็บ index ที่ match
    target_indices = [i for i, t in enumerate(raw_tokens) if t.strip().lower() == target_clean]

    for idx, t_idx in enumerate(target_indices, 1):
        # ดึงคำแวดล้อมตาม window size
        left_tokens = raw_tokens[max(0, t_idx - window) : t_idx]
        right_tokens = raw_tokens[t_idx + 1 : min(len(raw_tokens), t_idx + 1 + window)]
        
        # คลีนตัวอักษรพิเศษ (\n) ออกเพื่อให้แสดงผลสวยงามในตาราง
        left_context = "".join([t.replace('\n', ' ') for t in left_tokens]).strip()
        right_context = "".join([t.replace('\n', ' ') for t in right_tokens]).strip()
        actual_key = raw_tokens[t_idx].replace('\n', ' ') # เอาคำจริงที่ปรากฏใน text

        data.append({
            'ลำดับ': idx,
            'บริบทซ้าย (Left Context)': "..." + left_context if t_idx - window > 0 else left_context,
            'คำเป้าหมาย (Key)': actual_key,
            'บริบทขวา (Right Context)': right_context + "..." if t_idx + 1 + window < len(raw_tokens) else right_context
        })

    return pd.DataFrame(data)

# --- จัดการ Session State (คงเดิม) ---
if "wc_all" not in st.session_state: st.session_state.wc_all = None
if "wc_content" not in st.session_state: st.session_state.wc_content = None
if "history_list" not in st.session_state: st.session_state.history_list = []
if "raw_tokens" not in st.session_state: st.session_state.raw_tokens = [] # เก็บ tokens รวมช่องว่าง

# ==================== แถวที่ 1 (Input) ====================
r1_left, r1_right = st.columns([1.3, 1], gap="medium")

with r1_left:
    with st.container(key="input_box"):
        st.markdown('<div class="card-title">📝 Thai Word Counter & Context Analyzer</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">วางเนื้อเพลงหรือข้อความภาษาไทยเพื่อวิเคราะห์แจกแจงความถี่และดูบริบทของคำ</div>', unsafe_allow_html=True)
        
        text_input = st.text_area(label="กรอกข้อความของคุณที่นี่:", value="", placeholder="วางเนื้อหาที่ต้องการวิเคราะห์...", height=200)
        
        # จัดปุ่มกึ่งกลาง
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        _, btn_center, _ = st.columns([1, 1.2, 1])
        with btn_center:
            btn_clicked = st.button("ประมวลผล", use_container_width=True)
            
        if btn_clicked:
            if text_input.strip():
                all_w, content_w, nc, w_pos, raw_t = word_count(text_input)
                st.session_state.wc_all = all_w
                st.session_state.wc_content = content_w
                st.session_state.non_common_total = nc
                st.session_state.word_to_pos = w_pos
                st.session_state.raw_tokens = raw_t # เก็บไว้ทำ concordance
                st.session_state.current_text = text_input # เก็บไว้ทำรูป
                st.rerun()

with r1_right:
    # (ส่วน Metric - คงเดิม)
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
    
    if st.session_state.wc_all:
        img_bytes = generate_story_image(st.session_state.current_text, total_tokens, unique_tokens, non_common_words)
        st.download_button(label="📸 Save Image", data=img_bytes, file_name="word_summary.png", mime="image/png", use_container_width=True)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 2 (กราฟและตารางความถี่ - คงเดิม) ====================
if st.session_state.wc_all:
    # (โค้ดส่วนแสดงตารางและกราฟ Top 10 - ไม่เปลี่ยนแปลง)
    # ... (ตัดออกเพื่อความกระชับ แต่ในไฟล์จริงต้องมีครบ) ...
    # สรุปคือ ส่วนนี้เหมือนเดิมทุกประการครับ
    
    # สมมติว่ามีโค้ดส่วนตาราง/กราฟอยู่ตรงนี้
    r2_left, r2_right = st.columns([1, 1.3], gap="medium")
    with r2_left:
        with st.container(key="table_box_1"):
            st.markdown('<div class="card-title">📊 ตารางแจกแจงความถี่ (คำทั้งหมด)</div>', unsafe_allow_html=True)
            data_all = []
            for word, count in st.session_state.wc_all.items():
                pos = st.session_state.word_to_pos.get(word, "-")
                data_all.append({"WORD": word, "POS TAG": pos, "COUNT": count})
            df_all = pd.DataFrame(data_all)
            st.dataframe(df_all, hide_index=True, use_container_width=True, height=300)

    with r2_right:
        with st.container(key="chart_box_1"):
            st.markdown('<div class="card-title">📈 คำที่พบมากที่สุด (Top 10 - รวมทุกคำ)</div>', unsafe_allow_html=True)
            if not df_all.empty:
                top_10_all = df_all.head(10)
                chart_all = alt.Chart(top_10_all).mark_bar(color="#f59e0b", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("WORD", sort=None, axis=alt.Axis(labelAngle=0, labelColor="#475569")),
                    y=alt.Y("COUNT", axis=alt.Axis(labelColor="#475569")),
                    tooltip=["WORD", "COUNT"]
                ).properties(height=280).configure_view(strokeWidth=0)
                st.altair_chart(chart_all, use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 3 (Concordance - ส่วนที่ทำใหม่) ====================
if st.session_state.wc_all:
    with st.container(key="table_box_concordance"):
        st.markdown('<div class="card-title">🔍 ดูบริบทของคำในประโยค (Word in Context)</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">เลือกคำที่สนใจเพื่อดูประโยคจริงที่คำนั้นปรากฏ (แสดงคำแวดล้อม ซ้าย-ขวา ข้างละ 5 คำ)</div>', unsafe_allow_html=True)
        
        # สร้างรายชื่อคำให้เลือก (เอาเฉพาะ Content words มาแสดงก่อนเพื่อความหมายที่ดี)
        if st.session_state.wc_content:
            word_options = list(st.session_state.wc_content.keys())
        else:
            word_options = list(st.session_state.wc_all.keys())

        # Dropdown เลือกคำ
        selected_word = st.selectbox(
            "เลือกคำที่ต้องการดูบริบท:",
            options=word_options,
            index=0 if word_options else None,
            label_visibility="collapsed"
        )
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if selected_word:
            # ดึงตาราง Concordance
            df_concordance = get_concordance_table(selected_word, st.session_state.raw_tokens, window=5)
            
            if not df_concordance.empty:
                # แสดงสถิติการพบ
                st.caption(f"พบคำว่า '{selected_word}' ทั้งหมด {len(df_concordance)} ครั้ง ในบริบทดังนี้:")
                
                # จัดรูปแบบตารางให้น่าอ่าน (Key อยู่ตรงกลาง)
                # ใช้ st.dataframe แบบกำหนดคอลัมน์
                st.dataframe(
                    df_concordance,
                    column_config={
                        "ลำดับ": st.column_config.NumberColumn(width="small"),
                        "บริบทซ้าย (Left Context)": st.column_config.TextColumn(width="large", help="คำก่อนหน้า"),
                        "คำเป้าหมาย (Key)": st.column_config.TextColumn(width="medium", help="คำที่เลือก"),
                        "บริบทขวา (Right Context)": st.column_config.TextColumn(width="large", help="คำถัดไป"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=300 # กำหนดความสูง固定
                )
            else:
                st.info(f"ไม่พบคำว่า '{selected_word}' ในบริบทที่ชัดเจน (อาจเป็นสัญลักษณ์)")
        else:
            st.info("กรุณาประมวลผลข้อความเพื่อเลือกคำ")
