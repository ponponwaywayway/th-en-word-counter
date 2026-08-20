import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import operator
import io
import base64
import altair as alt
from PIL import Image, ImageDraw, ImageFont
import urllib.request
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

# --- ดึง Dataset ประโยคภาษาไทยขนาดใหญ่ + Brown English Corpus พร้อมสร้าง Inverted Index ---
@st.cache_resource(show_spinner=False)
def build_multi_corpus_index():
    index_eng = {}
    index_thai = {}

    # 1. ภาษาอังกฤษ: NLTK Brown Corpus (57,000+ ประโยค)
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

    # 2. ภาษาไทย: ดึงชุดประโยคภาษาไทยขนาดใหญ่จาก Open Dataset
    thai_sentences = []
    data_urls = [
        "https://raw.githubusercontent.com/PyThaiNLP/th-cv-sentences/main/data/text.txt",
        "https://raw.githubusercontent.com/tatoeba/tatoeba-datasets/master/tha/tha_sentences.tsv"
    ]
    
    for url in data_urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=4) as response:
                content = response.read().decode('utf-8', errors='ignore')
                for line in content.splitlines()[:6000]:
                    parts = line.split('\t')
                    sent_text = parts[-1].strip() if len(parts) > 1 else line.strip()
                    if sent_text and len(sent_text) > 8:
                        thai_sentences.append(sent_text)
        except Exception:
            continue

    # ชุดประโยคเพิ่มเติมครอบคลุมคำในชีวิตประจำวัน
    master_thai_sents = [
        "ฉันยังคงรอเธออยู่ที่เดิมเสมอไม่ว่าเวลาจะผ่านไปนานแค่ไหน",
        "เขากำลังยืนรอรถเมล์อยู่ที่ป้ายหน้าโรงเรียนในตอนเย็น",
        "อย่าปล่อยให้ใครต้องรอนานเกินไปเพราะเวลามีค่าสำหรับทุกคน",
        "พวกเรานั่งรอฟังผลการประกาศรางวัลด้วยความตื่นเต้นอย่างมาก",
        "การรอคอยอย่างมีความหวังช่วยให้เรามีกำลังใจในการก้าวต่อไปข้างหน้า",
        "ความรักทำให้คนเรามีพลังในการใช้ชีวิตและสร้างสรรค์สิ่งดีงาม",
        "การพัฒนาเทคโนโลยีในปัจจุบันมีความก้าวหน้าอย่างรวดเร็วและต่อเนื่อง",
        "ดนตรีและศิลปะช่วยบำบัดจิตใจและสร้างความสุขให้กับผู้ฟังเสมอ",
        "แสงแดดยามเช้าส่องประกายผ่านม่านหมอกลงมาบนยอดดอยอย่างงดงาม",
        "การเดินทางท่องเที่ยวเปิดประสบการณ์ใหม่และสร้างความทรงจำที่ดี",
        "ความพยายามและการฝึกฝนอย่างสม่ำเสมอจะนำพาไปสู่ความสำเร็จ",
        "เราควรให้ความสำคัญกับการดูแลรักษาสิ่งแวดล้อมเพื่อคนรุ่นหลัง",
        "ความสุขที่แท้จริงเกิดจากความสงบในใจและการมองโลกในแง่ดี",
        "การอ่านหนังสือช่วยเปิดโลกทัศน์และเพิ่มพูนความรู้รอบตัวอยู่เสมอ",
        "รอยยิ้มและความจริงใจเป็นสิ่งที่มีค่าที่สุดในการสร้างมิตรภาพ",
        "กาลเวลาและประสบการณ์ทำให้เราเติบโตเป็นผู้ใหญ่ที่เข้มแข็ง",
        "สายลมหนาวพัดผ่านทุ่งหญ้าเขียวขจีในฤดูเก็บเกี่ยวของชาวบ้าน",
        "กำลังใจและความเชื่อมั่นเป็นสิ่งสำคัญในการก้าวข้ามผ่านอุปสรรค",
        "การออกกำลังกายและพักผ่อนให้เพียงพอช่วยเสริมสร้างสุขภาพร่างกาย",
        "ท้องฟ้ายามค่ำคืนเต็มไปด้วยดวงดาวระยิบระยับพร่างพราวทั่วทั้งฟ้า",
        "ความซื่อสัตย์เป็นหัวใจสำคัญของการทำงานร่วมกับผู้อื่นในสังคม",
        "อาหารไทยมีรสชาติกลมกล่อมและเป็นเอกลักษณ์ที่ได้รับความนิยมไปทั่วโลก",
        "ภาษาและวัฒนธรรมเป็นมรดกทางปัญญาที่สะท้อนถึงประวัติศาสตร์อันยาวนาน",
        "การฟังความคิดเห็นของผู้อื่นช่วยสร้างความเข้าใจและสันติสุขในสังคม",
        "เทคโนโลยีปัญญาประดิษฐ์กำลังเข้ามามีบทบาทสำคัญในชีวิตประจำวัน",
        "ดอกไม้บานสะพรั่งส่งกลิ่นหอมอบอวลไปทั่วสวนในยามเช้าตรู่",
        "ภาพยนตร์เรื่องนี้ถ่ายทอดเรื่องราวชีวิตได้อย่างลึกซึ้งและกินใจผู้ชม"
    ]
    all_thai = thai_sentences + master_thai_sents
    sym = {'"', '[', ']', '(', ')', ',', '!', '.', '\n', '\s', ' ', '', 'ๆ', '?', ':', "'", '“', '”', '%', '-', '–', '—', '\\', '/', '>', '<', ';', '+', '*', '&', '’', '‘'}

    for sent_str in all_thai:
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
                        "Thai Standard Corpus"
                    ))

    return index_eng, index_thai

INDEX_ENG_CORPUS, INDEX_THAI_CORPUS = build_multi_corpus_index()

# --- ค้นหา KWIC Concordance แบบ Hybrid (Corpus -> Fallback to Input Text) ---
def search_corpus_concordance_hybrid(target_word: str, raw_input_tokens: list, max_results: int = 10):
    if not target_word:
        return pd.DataFrame(columns=["ลำดับ", "บริบทซ้าย (Left Context)", "คำเป้าหมาย (Key)", "บริบทขวา (Right Context)", "คลังภาษา (Corpus)"])

    is_eng = is_english_word(target_word)
    rows = []
    
    # 1. ค้นหาใน Standard Corpus ก่อน
    if is_eng:
        target_clean = target_word.strip().lower()
        matches = INDEX_ENG_CORPUS.get(target_clean, [])
    else:
        target_clean = target_word.strip()
        matches = INDEX_THAI_CORPUS.get(target_clean, [])

    for idx, (left, key, right, corpus_name) in enumerate(matches[:max_results], start=1):
        rows.append({
            "ลำดับ": idx,
            "บริบทซ้าย (Left Context)": left,
            "คำเป้าหมาย (Key)": key,
            "บริบทขวา (Right Context)": right,
            "คลังภาษา (Corpus)": corpus_name
        })

    # 2. ถ้าใน Corpus ไม่มีผลลัพธ์ ให้ Fallback ดึงจากข้อความที่ผู้ใช้กรอก (Input Text Context)
    if not rows and raw_input_tokens:
        match_idx = 1
        t_compare = target_word.strip().lower() if is_eng else target_word.strip()
        
        for i, t in enumerate(raw_input_tokens):
            curr_val = t.strip().lower() if is_eng else t.strip()
            if curr_val == t_compare:
                left_tokens = raw_input_tokens[max(0, i - 5) : i]
                right_tokens = raw_input_tokens[i + 1 : min(len(raw_input_tokens), i + 6)]
                
                left = (" ".join(left_tokens) if is_eng else "".join(left_tokens)).replace('\n', ' ').strip()
                right = (" ".join(right_tokens) if is_eng else "".join(right_tokens)).replace('\n', ' ').strip()
                key = t.strip()
                
                rows.append({
                    "ลำดับ": match_idx,
                    "บริบทซ้าย (Left Context)": f"...{left}" if i > 0 and left else left,
                    "คำเป้าหมาย (Key)": key,
                    "บริบทขวา (Right Context)": f"{right}..." if (i + 1 < len(raw_input_tokens)) and right else right,
                    "คลังภาษา (Corpus)": "จากข้อความที่คุณกรอก (User Input Context)"
                })
                match_idx += 1
                if len(rows) >= max_results:
                    break

    return pd.DataFrame(rows)

# --- ฟังก์ชันสร้างภาพ 9:16 เพื่อแชร์ ---
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

    font_paths = [
        "tahoma.ttf", "leelawad.ttf", "Thonburi.ttc", "Angsana.ttc",
        "/System/Library/Fonts/Supplemental/Thonburi.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    font_main = None
    for p in font_paths:
        try:
            font_title = ImageFont.truetype(p, 54)
            font_sub = ImageFont.truetype(p, 32)
            font_body = ImageFont.truetype(p, 28)
            font_num = ImageFont.truetype(p, 64)
            font_main = True
            break
        except Exception:
            continue

    if not font_main:
        font_title = ImageFont.load_default()
        font_sub = font_title
        font_body = font_title
        font_num = font_title

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

# --- ฟังก์ชันตัดและนับคำ ---
def word_count(lyrics: str):
    if not lyrics.strip():
        return {}, {}, 0, {}, {}, []
    
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

    return dict(sorted_all_list), dict(sorted_content_list), non_common_total_count, word_to_pos, pos_dict_sorted, raw_tokens

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

if "raw_tokens" not in st.session_state:
    st.session_state.raw_tokens = []

if "history_list" not in st.session_state:
    st.session_state.history_list = []

if "current_text" not in st.session_state:
    st.session_state.current_text = ""

def apply_history():
    selected = st.session_state.selected_history
    if selected and selected != "-- เลือกดูประวัติข้อความเก่า --":
        st.session_state.current_text = selected
        all_w, content_w, nc, w_pos, p_dict, r_tokens = word_count(selected)
        st.session_state.wc_all = all_w
        st.session_state.wc_content = content_w
        st.session_state.non_common_total = nc
        st.session_state.word_to_pos = w_pos
        st.session_state.pos_dict_sorted = p_dict
        st.session_state.raw_tokens = r_tokens

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
                
                all_w, content_w, nc, w_pos, p_dict, r_tokens = word_count(text_input)
                st.session_state.wc_all = all_w
                st.session_state.wc_content = content_w
                st.session_state.non_common_total = nc
                st.session_state.word_to_pos = w_pos
                st.session_state.pos_dict_sorted = p_dict
                st.session_state.raw_tokens = r_tokens
                st.rerun()
            else:
                st.session_state.wc_all = None
                st.session_state.wc_content = None
                st.session_state.non_common_total = 0
                st.session_state.word_to_pos = {}
                st.session_state.pos_dict_sorted = None
                st.session_state.raw_tokens = []

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

# ==================== แถวที่ 5 (ตารางส่องบริบทประโยคจริง: Hybrid KWIC Concordance) ====================
with st.container(key="table_box_corpus_kwic"):
    st.markdown('<div class="card-title">📚 ตัวอย่างประโยคจริงจากการใช้งานทั่วไป (Corpus KWIC Concordance)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">เลือกคำเพื่อดูบริบทประโยคจริงในการใช้งานทั่วไป (ไทย: Standard Thai Corpus | อังกฤษ: Brown Corpus)</div>', unsafe_allow_html=True)
    
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
        
        df_corpus_kwic = search_corpus_concordance_hybrid(
            target_word=selected_target_word,
            raw_input_tokens=st.session_state.raw_tokens,
            max_results=10
        )
        
        if not df_corpus_kwic.empty:
            st.dataframe(
                df_corpus_kwic[["ลำดับ", "บริบทซ้าย (Left Context)", "คำเป้าหมาย (Key)", "บริบทขวา (Right Context)", "คลังภาษา (Corpus)"]],
                hide_index=True,
                use_container_width=True,
                height=280
            )
        else:
            st.info(f"ไม่พบตัวอย่างประโยคของคำว่า '{selected_target_word}'")
    else:
        st.markdown("<p style='color: #8a8ca3; height: 120px; display: flex; align-items: center; justify-content: center;'>ยังไม่มีข้อมูลการแสดงผล</p>", unsafe_allow_html=True)
