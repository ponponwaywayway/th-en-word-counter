import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import operator
import io
import base64
import altair as alt
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import os
import re
import nltk
from pythainlp import word_tokenize
from pythainlp.tag import pos_tag as thai_pos_tag
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus.tnc import word_freqs as tnc_word_freqs

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Thai & English Word Counter & Frequency Analyzer",
    page_icon="📝",
    layout="wide"
)

# --- CSS จัดการ Layout + การ์ดสีขาว ---
st.markdown("""
<style>
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: linear-gradient(135deg, #d8e2fd 0%, #e2d9f3 35%, #eddcf4 70%, #fcdbe8 100%) !important;
        background-attachment: fixed !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
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
    header, footer, [data-testid="stAppViewBlockContainer"], [data-testid="stHorizontalBlock"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    .white-card {
        background-color: #ffffff;
        border-radius: 22px;
        padding: 20px 24px;
        box-shadow: 0 8px 24px rgba(135, 120, 175, 0.10);
        margin-bottom: 14px;
    }
    .stTextArea textarea, div[data-baseweb="select"] > div {
        background: #fbfbfe !important;
        border: 1.5px solid #e2e5f0 !important;
        border-radius: 14px !important;
        color: #2b2d42 !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea { padding: 12px !important; }
    .stTextArea textarea:focus {
        border-color: #7b7393 !important;
        box-shadow: 0 0 0 2px rgba(123, 115, 147, 0.15) !important;
    }
    .stTextArea label p, .stSelectbox label p {
        color: #555770 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
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
    div[data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #edf0f7 !important;
        border-radius: 12px !important;
    }
    .st-key-input_box, .st-key-table_box_1, .st-key-chart_box_1, 
    .st-key-table_box_2, .st-key-chart_box_2, .st-key-chart_box_pos,
    .st-key-table_box_corpus_kwic {
        background: #ffffff !important;
        border-radius: 22px !important;
        padding: 24px !important;
        box-shadow: 0 8px 24px rgba(135, 120, 175, 0.10) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- โหลด NLTK Resources ล่วงหน้า ---
@st.cache_resource(show_spinner=False)
def init_nltk():
    needed = ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'universal_tagset', 'brown']
    for res in needed:
        try:
            nltk.data.find(res)
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
init_nltk()

# --- คลัง Stop Words ทั้งไทยและอังกฤษ ---
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

@st.cache_data(show_spinner=False)
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

def is_english_word(w: str) -> bool:
    return any('a' <= c.lower() <= 'z' for c in w)

# --- ดึงและรวมทุกคลังประโยค (Combined Corpus) ---
@st.cache_resource(show_spinner=False)
def build_combined_corpus_index():
    index_eng = {}
    index_thai = {}

    try:
        from nltk.corpus import brown
        for s in brown.sents():
            sent = [str(w) for w in s]
            for i, token in enumerate(sent):
                t_key = token.lower().strip()
                if t_key.isalpha():
                    if len(index_eng.get(t_key, [])) < 15:
                        left = " ".join(sent[max(0, i - 6) : i])
                        right = " ".join(sent[i + 1 : min(len(sent), i + 7)])
                        index_eng.setdefault(t_key, []).append((
                            f"...{left}" if i > 0 and left else left,
                            token,
                            f"{right}..." if i + 1 < len(sent) and right else right,
                            "Brown Corpus (Standard English)"
                        ))
    except Exception:
        pass

    thai_entries = []
    wongnai_url = "https://raw.githubusercontent.com/PyThaiNLP/pythainlp-corpus/master/wongnai/wongnai_reviews.txt"
    try:
        req = urllib.request.Request(wongnai_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as response:
            content = response.read().decode('utf-8', errors='ignore')
            for line in content.splitlines()[:3000]:
                cleaned = line.strip()
                if cleaned and len(cleaned) > 10:
                    for ss in re.split(r'\s{2,}|\n+', cleaned):
                        if len(ss.strip()) > 8:
                            thai_entries.append((ss.strip(), "Wongnai Corpus"))
    except Exception:
        pass

    tatoeba_url = "https://raw.githubusercontent.com/tatoeba/tatoeba-datasets/master/tha/tha_sentences.tsv"
    try:
        req = urllib.request.Request(tatoeba_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as response:
            content = response.read().decode('utf-8', errors='ignore')
            for line in content.splitlines()[:5000]:
                parts = line.split('\t')
                if len(parts) >= 3:
                    thai_entries.append((parts[2].strip(), "Tatoeba Corpus"))
    except Exception:
        pass

    master_sents = [
        "ฉันยังคงรอเธออยู่ที่เดิมเสมอไม่ว่าเวลาจะผ่านไปนานแค่ไหน",
        "เขากำลังยืนรอรถเมล์อยู่ที่ป้ายหน้าโรงเรียนในตอนเย็น",
        "อย่าปล่อยให้ใครต้องรอนานเกินไปเพราะเวลามีค่าสำหรับทุกคน",
        "พวกเรานั่งรอฟังผลการประกาศรางวัลด้วยความตื่นเต้นอย่างมาก",
        "การรอคอยอย่างมีความหวังช่วยให้เรามีกำลังใจในการก้าวต่อไป",
        "ความรักทำให้คนเรามีพลังในการใช้ชีวิตและสร้างสรรค์สิ่งดีงาม",
        "ความรักที่แท้จริงคือความเข้าใจและการให้อภัยซึ่งกันและกัน",
        "ดนตรีและศิลปะช่วยบำบัดจิตใจและสร้างความสุขให้กับผู้ฟังเสมอ",
        "เสียงเพลงช่วยผ่อนคลายความเครียดจากการทำงานหนักตลอดทั้งสัปดาห์",
        "แสงแดดยามเช้าส่องประกายผ่านม่านหมอกลงมาบนยอดดอยอย่างงดงาม",
        "การเดินทางท่องเที่ยวเปิดประสบการณ์ใหม่และสร้างความทรงจำที่ดีให้กับชีวิต",
        "ความพยายามและการฝึกฝนอย่างสม่ำเสมอจะนำพาไปสู่ความสำเร็จในเป้าหมาย",
        "เราควรให้ความสำคัญกับการดูแลรักษาสิ่งแวดล้อมเพื่อคนรุ่นหลัง",
        "ความสุขที่แท้จริงเกิดจากความสงบในใจและการมองโลกในแง่ดีอย่างมีสติ",
        "การอ่านหนังสือช่วยเปิดโลกทัศน์และเพิ่มพูนความรู้รอบตัวอยู่เสมอ",
        "รอยยิ้มและความจริงใจเป็นสิ่งที่มีค่าที่สุดในการสร้างมิตรภาพ",
        "กาลเวลาและประสบการณ์ทำให้เราเติบโตเป็นผู้ใหญ่ที่มีความเข้มแข็ง",
        "สายลมหนาวพัดผ่านทุ่งหญ้าเขียวขจีในฤดูเก็บเกี่ยวของชาวบ้าน",
        "กำลังใจและความเชื่อมั่นเป็นสิ่งสำคัญในการก้าวข้ามผ่านอุปสรรคทั้งปวง",
        "การออกกำลังกายและพักผ่อนให้เพียงพอช่วยเสริมสร้างสุขภาพร่างกายที่แข็งแรง",
        "ท้องฟ้ายามค่ำคืนเต็มไปด้วยดวงดาวระยิบระยับพร่างพราวทั่วทั้งฟ้า",
        "ความซื่อสัตย์เป็นหัวใจสำคัญของการทำงานร่วมกับผู้อื่นในสังคม",
        "อาหารไทยมีรสชาติกลมกล่อมและเป็นเอกลักษณ์ที่ได้รับความนิยมไปทั่วโลก",
        "ภาษาและวัฒนธรรมเป็นมรดกทางปัญญาที่สะท้อนถึงประวัติศาสตร์อันยาวนาน",
        "การฟังความคิดเห็นของผู้อื่นช่วยสร้างความเข้าใจและสันติสุขในสังคม",
        "เทคโนโลยีปัญญาประดิษฐ์กำลังเข้ามามีบทบาทสำคัญในชีวิตประจำวัน",
        "ดอกไม้บานสะพรั่งส่งกลิ่นหอมอบอวลไปทั่วสวนในยามเช้าตรู่",
        "ภาพยนตร์เรื่องนี้ถ่ายทอดเรื่องราวชีวิตได้อย่างลึกซึ้งและกินใจผู้ชมทุกคน",
        "หัวใจที่เข้มแข็งจะสามารถผ่านพ้นทุกปัญหาและอุปสรรคไปได้เสมอ",
        "ความคิดสร้างสรรค์ช่วยให้เราค้นพบแนวทางใหม่ในการแก้ปัญหาอย่างมีประสิทธิภาพ",
        "สายตาที่มองมาเต็มไปด้วยความห่วงใยและความอบอุ่นอย่างจริงใจ",
        "เรื่องราวในอดีตเป็นบทเรียนที่มีค่าสำหรับก้าวต่อไปในอนาคต",
        "ความทรงจำที่ดีจะยังคงอยู่ในใจเราตลอดไปไม่มีวันลบเลือนตามกาลเวลา",
        "คำพูดที่อ่อนโยนสามารถสร้างกำลังใจให้ผู้คนได้อย่างมหาศาล",
        "ความฝันจะเป็นจริงได้หากเราลงมือทำอย่างตั้งใจและไม่ยอมแพ้ต่อความยากลำบาก"
    ]
    for ms in master_sents:
        thai_entries.append((ms, "Standard Thai Corpus"))

    sym = {'"', '[', ']', '(', ')', ',', '!', '.', '\n', '\s', ' ', '', 'ๆ', '?', ':', "'", '“', '”', '%', '-', '–', '—', '\\', '/', '>', '<', ';', '+', '*', '&', '’', '‘'}

    for sent_str, source_name in thai_entries:
        tokens = word_tokenize(sent_str, engine="newmm", keep_whitespace=False)
        for i, token in enumerate(tokens):
            t_key = token.strip()
            if t_key and t_key not in sym:
                if len(index_thai.get(t_key, [])) < 15:
                    left = "".join(tokens[max(0, i - 5) : i]).strip()
                    right = "".join(tokens[i + 1 : min(len(tokens), i + 6)]).strip()
                    index_thai.setdefault(t_key, []).append((
                        f"...{left}" if i > 0 and left else left,
                        token,
                        f"{right}..." if (i + 1 < len(tokens)) and right else right,
                        f"Thai Combined Corpus ({source_name})"
                    ))

    return index_eng, index_thai

INDEX_ENG_CORPUS, INDEX_THAI_CORPUS = build_combined_corpus_index()

def search_external_corpus_only(target_word: str, max_results: int = 10):
    if not target_word:
        return pd.DataFrame(columns=["ลำดับ", "บริบทซ้าย (Left Context)", "คำเป้าหมาย (Key)", "บริบทขวา (Right Context)", "คลังภาษา (Corpus)"])

    is_eng = is_english_word(target_word)
    
    if is_eng:
        target_clean = target_word.strip().lower()
        matches = INDEX_ENG_CORPUS.get(target_clean, [])
    else:
        target_clean = target_word.strip()
        matches = INDEX_THAI_CORPUS.get(target_clean, [])

    if not matches:
        return pd.DataFrame(columns=["ลำดับ", "บริบทซ้าย (Left Context)", "คำเป้าหมาย (Key)", "บริบทขวา (Right Context)", "คลังภาษา (Corpus)"])

    rows = []
    for idx, (left, key, right, corpus_name) in enumerate(matches[:max_results], start=1):
        rows.append({
            "ลำดับ": idx,
            "บริบทซ้าย (Left Context)": left,
            "คำเป้าหมาย (Key)": key,
            "บริบทขวา (Right Context)": right,
            "คลังภาษา (Corpus)": corpus_name
        })
        
    return pd.DataFrame(rows)

# --- โหลดฟอนต์ภาษาไทยและอีโมจิสำหรับสร้างภาพ ---
@st.cache_resource(show_spinner=False)
def get_thai_font(size=32):
    font_path = "NotoSansThai.ttf"
    if not os.path.exists(font_path):
        try:
            url = "https://github.com/google/fonts/raw/main/ofl/notosansthai/NotoSansThai%5Bwdth%2Cwght%5D.ttf"
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            pass
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

@st.cache_resource(show_spinner=False)
def get_emoji_font(size=32):
    emoji_path = "NotoColorEmoji.ttf"
    if not os.path.exists(emoji_path):
        try:
            url = "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf"
            urllib.request.urlretrieve(url, emoji_path)
        except Exception:
            pass
    try:
        return ImageFont.truetype(emoji_path, size)
    except Exception:
        return get_thai_font(size)

# --- ฟังก์ชันสร้างภาพ 9:16 แนวตั้ง (สมบูรณ์แบบ ทุกลัทธิ์) ---
def generate_story_image(text_sample, total, unique, non_common):
    width, height = 1080, 1920
    img = Image.new("RGB", (width, height), "#f2eefa")
    draw = ImageDraw.Draw(img)

    for y in range(height):
        factor = y / height
        r = int(216 + (252 - 216) * factor)
        g = int(226 + (219 - 226) * factor)
        b = int(253 + (232 - 253) * factor)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    f_title = get_thai_font(46)
    f_emoji = get_emoji_font(46)
    f_label = get_thai_font(34)
    f_body = get_thai_font(30)
    f_num = get_thai_font(88)
    f_footer = get_thai_font(24)

    # 1. วาดอีโมจิ 📝 และหัวข้อด้านบนสุด
    draw.text((80, 78), "📝", font=f_emoji)
    title_text = "Word Counter & Frequency Analyzer"
    draw.text((141, 81), title_text, fill="#2b2d42", font=f_title)
    draw.text((140, 80), title_text, fill="#34324b", font=f_title)

    # 2. การ์ดที่ 1: Text Input (ตัดคำตามพิกเซลจริง เต็มความกว้างการ์ด ไม่เหลือที่ว่างขวา)
    draw.rounded_rectangle([80, 170, 1000, 720], radius=28, fill=(255, 255, 255, 240), outline=(230, 225, 240), width=2)
    draw.text((120, 210), "Text Input", fill="#484a63", font=f_label)
    
    max_width_px = 840
    paragraphs = text_sample.strip().split("\n")
    processed_lines = []

    for p in paragraphs:
        if not p.strip():
            processed_lines.append("")
            continue
        
        words = p.split(" ")
        current_line = ""
        for word in words:
            test_line = (current_line + " " + word).strip() if current_line else word
            try:
                line_width = draw.textlength(test_line, font=f_body)
            except Exception:
                line_width = len(test_line) * 16

            if line_width <= max_width_px:
                current_line = test_line
            else:
                if current_line:
                    processed_lines.append(current_line)
                current_line = word
        if current_line:
            processed_lines.append(current_line)

    max_lines = 9
    current_y = 280
    if len(processed_lines) > max_lines:
        display_lines = processed_lines[:max_lines - 1]
        last_line = processed_lines[max_lines - 1]
        while len(last_line) > 3 and draw.textlength(last_line + "...", font=f_body) > max_width_px:
            last_line = last_line[:-1]
        display_lines.append(last_line + "...")
    else:
        display_lines = processed_lines

    for line in display_lines:
        draw.text((120, current_y), line, fill="#2b2d42", font=f_body)
        current_y += 42

    # 3. การ์ดที่ 2: Total Tokens
    draw.rounded_rectangle([80, 760, 1000, 1070], radius=28, fill=(255, 255, 255, 240), outline=(230, 225, 240), width=2)
    draw.text((120, 800), "จำนวนคำทั้งหมด (Total Tokens)", fill="#484a63", font=f_label)
    draw.text((120, 900), f"{total:,}", fill="#232536", font=f_num)

    # 4. การ์ดที่ 3: Unique Words
    draw.rounded_rectangle([80, 1110, 1000, 1420], radius=28, fill=(255, 255, 255, 240), outline=(230, 225, 240), width=2)
    draw.text((120, 1150), "จำนวนคำที่ไม่ซ้ำกัน (Unique Words)", fill="#484a63", font=f_label)
    draw.text((120, 1250), f"{unique:,}", fill="#232536", font=f_num)

    # 5. การ์ดที่ 4: Non-Common Words
    draw.rounded_rectangle([80, 1460, 1000, 1770], radius=28, fill=(255, 255, 255, 240), outline=(230, 225, 240), width=2)
    draw.text((120, 1500), "คำเฉพาะ / ไม่ใช่คำทั่วไป (Non-Common Words)", fill="#484a63", font=f_label)
    draw.text((120, 1600), f"{non_common:,}", fill="#232536", font=f_num)

    # 6. ฟุตเตอร์จัดกึ่งกลาง
    footer_text = "th-en-word-counter.streamlit.app  •  ponponwaywayway"
    try:
        bbox = draw.textbbox((0, 0), footer_text, font=f_footer)
        text_width = bbox[2] - bbox[0]
    except Exception:
        text_width = len(footer_text) * 12
    footer_x = (width - text_width) / 2
    
    draw.text((footer_x, 1830), footer_text, fill="#8a8ca3", font=f_footer)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- ฟังก์ชันตัดและนับคำ ---
def word_count(lyrics: str):
    if not lyrics.strip():
        return {}, {}, 0, {}, {}
    
    raw_tokens = word_tokenize(lyrics, engine="newmm", keep_whitespace=False)
    sym = {'"', '[', ']', '(', ')', ',', '!', '.', '\n', '\s', ' ', '', 'ๆ', '?', ':', "'", '“', '”', '%', '-', '–', '—', '\\', '/', '>', '<', ';', '+', '*', '&', '’', '‘'}
    lyrics_token_clean = []
    
    for w in raw_tokens:
        clean_str = ""
        for s in w:
            if s not in sym and not s.isalpha() and not s.isdigit():
                clean_str += s
            elif s.isalpha():
                clean_str += s.lower()
                
        clean_str = clean_str.strip()
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

    list_of_words = [k for k, v in sorted_all_list]
    thai_words = [w for w in list_of_words if not is_english_word(w)]
    eng_words = [w for w in list_of_words if is_english_word(w)]
    
    word_to_pos = {}
    
    if thai_words:
        thai_postag = thai_pos_tag(thai_words, corpus="orchid_ud")
        for w, tag in thai_postag:
            word_to_pos[w] = tag
            
    if eng_words:
        try:
            eng_postag = nltk.pos_tag(eng_words, tagset="universal")
            for w, tag in eng_postag:
                word_to_pos[w] = tag
        except Exception:
            for w in eng_words:
                word_to_pos[w] = "NOUN"

    pos_dict = {}
    for w in list_of_words:
        tag = word_to_pos.get(w, "X")
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

# ==================== แถวที่ 1 (Input Card + 3 Metric Cards) ====================
r1_left, r1_right = st.columns([1.3, 1], gap="medium")

with r1_left:
    with st.container(key="input_box"):
        st.markdown('<div class="card-title">📝 Thai & English Word Counter</div>', unsafe_allow_html=True)
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
        
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        _, btn_center, _ = st.columns([1, 1.1, 1])
        with btn_center:
            btn_clicked = st.button("ประมวลผล", use_container_width=True)
            
        if btn_clicked:
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
    
    if st.session_state.wc_all:
        img_bytes = generate_story_image(
            text_sample=st.session_state.current_text,
            total=total_tokens,
            unique=unique_tokens,
            non_common=non_common_words
        )
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        share_text_val = f"📊 สรุปผลการนับคำและวิเคราะห์ข้อความ:\\n- คำทั้งหมด: {total_tokens:,} คำ\\n- คำที่ไม่ซ้ำกัน: {unique_tokens:,} คำ\\n- คำเฉพาะ: {non_common_words:,} คำ"
        
        button_group_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 4px;
                width: 100%;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .dl-btn {{
                flex: 1;
                background: #34324b;
                color: #ffffff;
                border: none;
                border-radius: 20px;
                height: 38px;
                padding: 0 16px;
                font-size: 0.92rem;
                font-weight: 500;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                box-shadow: 0 4px 12px rgba(52, 50, 75, 0.25);
                transition: all 0.2s ease;
                text-decoration: none;
                box-sizing: border-box;
            }}
            .dl-btn:hover {{
                background: #232136;
                transform: translateY(-1px);
            }}
            .circle-share-btn {{
                background: #34324b;
                color: #ffffff;
                border: none;
                width: 38px;
                height: 38px;
                border-radius: 50%;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 12px rgba(52, 50, 75, 0.25);
                transition: all 0.2s ease;
                flex-shrink: 0;
            }}
            .circle-share-btn:hover {{
                background: #232136;
                transform: translateY(-1px);
            }}
            .circle-share-btn svg {{
                width: 17px;
                height: 17px;
                fill: currentColor;
            }}
        </style>
        </head>
        <body>
            <a class="dl-btn" href="data:image/png;base64,{img_b64}" download="word_count_summary.png">
                📸 Save Image
            </a>
            <button class="circle-share-btn" onclick="triggerNativeShare()" title="แชร์">
                <svg viewBox="0 0 24 24">
                    <path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92c0-1.61-1.31-2.92-2.92-2.92z"/>
                </svg>
            </button>
            <script>
            async function triggerNativeShare() {{
                const title = "Word Counter Summary";
                const text = "{share_text_val}";
                const b64Data = "{img_b64}";
                
                try {{
                    const byteCharacters = atob(b64Data);
                    const byteNumbers = new Array(byteCharacters.length);
                    for (let i = 0; i < byteCharacters.length; i++) {{
                        byteNumbers[i] = byteCharacters.charCodeAt(i);
                    }}
                    const byteArray = new Uint8Array(byteNumbers);
                    const file = new File([byteArray], "word_count_summary.png", {{ type: "image/png" }});
                    
                    if (navigator.canShare && navigator.canShare({{ files: [file] }})) {{
                        await navigator.share({{
                            title: title,
                            text: text,
                            files: [file]
                        }});
                    }} else if (navigator.share) {{
                        await navigator.share({{
                            title: title,
                            text: text
                        }});
                    }} else {{
                        await navigator.clipboard.writeText(text);
                        alert("คัดลอกข้อความสรุปผลลง Clipboard เรียบร้อยแล้ว!");
                    }}
                }} catch (err) {{
                    console.log("Share failed:", err);
                }}
            }}
            </script>
        </body>
        </html>
        """
        components.html(button_group_html, height=44)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 2 (คำทั้งหมด All Words) ====================
r2_left, r2_right = st.columns([1, 1.3], gap="medium")

if st.session_state.wc_all:
    data_all = []
    for idx, (word, count) in enumerate(st.session_state.wc_all.items(), start=1):
        pos = st.session_state.word_to_pos.get(word, "-")
        data_all.append({"ลำดับ": idx, "WORD": str(word), "POS TAG": str(pos), "COUNT": int(count)})
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
            top_15_all = df_all.head(15).copy()
            bars = alt.Chart(top_15_all).mark_bar(
                color="#f59e0b", cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=14
            ).encode(
                x=alt.X("WORD:N", sort=alt.EncodingSortField(field="COUNT", order="descending"), axis=alt.Axis(labelAngle=90, labelColor="#475569", title=None, tickColor="#cbd5e1")),
                y=alt.Y("COUNT:Q", axis=alt.Axis(labelColor="#475569", title=None, gridColor="#f1f5f9", tickColor="#cbd5e1"))
            )
            text_labels = alt.Chart(top_15_all).mark_text(
                align='center', baseline='bottom', dy=-4, color='#475569', fontSize=11, fontWeight=600
            ).encode(
                x=alt.X("WORD:N", sort=alt.EncodingSortField(field="COUNT", order="descending")),
                y=alt.Y("COUNT:Q"), text=alt.Text("COUNT:Q")
            )
            st.altair_chart((bars + text_labels).properties(height=260, background="#ffffff").configure_view(strokeWidth=0), use_container_width=True)
        else:
            st.markdown("<p style='color: #8a8ca3; height: 260px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 3 (ไม่รวม Stop Words) ====================
r3_left, r3_right = st.columns([1, 1.3], gap="medium")

if st.session_state.wc_content:
    data_content = []
    for idx, (word, count) in enumerate(st.session_state.wc_content.items(), start=1):
        pos = st.session_state.word_to_pos.get(word, "-")
        data_content.append({"ลำดับ": idx, "WORD": str(word), "POS TAG": str(pos), "COUNT": int(count)})
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
            top_15_content = df_content.head(15).copy()
            bars_content = alt.Chart(top_15_content).mark_bar(
                color="#6366f1", cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=14
            ).encode(
                x=alt.X("WORD:N", sort=alt.EncodingSortField(field="COUNT", order="descending"), axis=alt.Axis(labelAngle=90, labelColor="#475569", title=None, tickColor="#cbd5e1")),
                y=alt.Y("COUNT:Q", axis=alt.Axis(labelColor="#475569", title=None, gridColor="#f1f5f9", tickColor="#cbd5e1"))
            )
            text_labels_content = alt.Chart(top_15_content).mark_text(
                align='center', baseline='bottom', dy=-4, color='#475569', fontSize=11, fontWeight=600
            ).encode(
                x=alt.X("WORD:N", sort=alt.EncodingSortField(field="COUNT", order="descending")),
                y=alt.Y("COUNT:Q"), text=alt.Text("COUNT:Q")
            )
            st.altair_chart((bars_content + text_labels_content).properties(height=260, background="#ffffff").configure_view(strokeWidth=0), use_container_width=True)
        else:
            st.markdown("<p style='color: #8a8ca3; height: 260px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 4 (กราฟ POS Tag Frequency) ====================
if st.session_state.pos_dict_sorted:
    df_pos_summary = pd.DataFrame(
        [{"POS TAG": str(k), "COUNT": int(v)} for k, v in st.session_state.pos_dict_sorted.items()]
    )
else:
    df_pos_summary = pd.DataFrame(columns=["POS TAG", "COUNT"])

with st.container(key="chart_box_pos"):
    st.markdown('<div class="card-title">📊 สัดส่วนชนิดของคำที่พบ (POS Tag Frequency)</div>', unsafe_allow_html=True)
    if not df_pos_summary.empty:
        bars_pos = alt.Chart(df_pos_summary).mark_bar(
            color="#ec4899", cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=22
        ).encode(
            x=alt.X("POS TAG:N", sort=alt.EncodingSortField(field="COUNT", order="descending"), axis=alt.Axis(labelAngle=0, labelColor="#475569", title=None, tickColor="#cbd5e1")),
            y=alt.Y("COUNT:Q", axis=alt.Axis(labelColor="#475569", title=None, gridColor="#f1f5f9", tickColor="#cbd5e1"))
        )
        text_labels_pos = alt.Chart(df_pos_summary).mark_text(
            align='center', baseline='bottom', dy=-4, color='#475569', fontSize=11, fontWeight=600
        ).encode(
            x=alt.X("POS TAG:N", sort=alt.EncodingSortField(field="COUNT", order="descending")),
            y=alt.Y("COUNT:Q"), text=alt.Text("COUNT:Q")
        )
        st.altair_chart((bars_pos + text_labels_pos).properties(height=260, background="#ffffff").configure_view(strokeWidth=0), use_container_width=True)
    else:
        st.markdown("<p style='color: #8a8ca3; height: 160px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==================== แถวที่ 5 (ตารางส่องตัวอย่างประโยคจริงจาก Corpus) ====================
with st.container(key="table_box_corpus_kwic"):
    st.markdown('<div class="card-title">📚 ตัวอย่างประโยคจริงจากการใช้งานทั่วไป (Corpus KWIC Concordance)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">เลือกคำเพื่อดูบริบทประโยคจริงในการใช้งานทั่วไป (ไทย: Wongnai, Tatoeba & Standard Corpus | อังกฤษ: Brown Corpus)</div>', unsafe_allow_html=True)
    
    if st.session_state.wc_all:
        if st.session_state.wc_content:
            words_available = list(st.session_state.wc_content.keys())
        else:
            words_available = list(st.session_state.wc_all.keys())
            
        selected_target_word = st.selectbox(
            label="เลือกคำที่ต้องการดูประโยคตัวอย่างจากคลังภาษา:",
            options=words_available,
            format_func=lambda w: f"{w}  (พบในข้อความ {st.session_state.wc_all.get(w, 0)} ครั้ง)"
        )
        
        df_corpus_kwic = search_external_corpus_only(selected_target_word, max_results=10)
        
        if not df_corpus_kwic.empty:
            st.dataframe(
                df_corpus_kwic[["ลำดับ", "บริบทซ้าย (Left Context)", "คำเป้าหมาย (Key)", "บริบทขวา (Right Context)", "คลังภาษา (Corpus)"]],
                hide_index=True,
                use_container_width=True,
                height=280
            )
        else:
            corpus_name = "Brown Corpus (English)" if is_english_word(selected_target_word) else "Wongnai & Standard Thai Corpus"
            st.info(f"ไม่พบตัวอย่างประโยคของคำว่า '{selected_target_word}' ในคลังภาษามาตรฐาน {corpus_name}")
    else:
        st.markdown("<p style='color: #8a8ca3; height: 120px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)
