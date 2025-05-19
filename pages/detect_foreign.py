import streamlit as st
import pymupdf
import pdfplumber
import pandas as pd
import fitz  # PyMuPDF
import re
import string
from wordfreq import word_frequency
from collections import Counter
from lingua import LanguageDetectorBuilder
import requests
import time  # Import the time module

# --- API Key ---
G_API_KEY = st.secrets.get("GOOGLE_TRANSLATE_API_KEY")
if not G_API_KEY:
    st.error("Google Translate API key not found in Streamlit secrets.")
    st.stop()

# --- Helper Functions ---
def dehyphenate_text(text):
    return re.sub(r'-\s*\n?\s*', '', text)

def extract_blocks_and_tables(pdf_path):
    start_time = time.time()
    doc = fitz.open(pdf_path)
    plumber_pdf = pdfplumber.open(pdf_path)
    blocks_data = []
    for page_num in range(len(doc)):
        page_fitz = doc.load_page(page_num)
        page_plumber = plumber_pdf.pages[page_num]

        # First: extract regular text blocks
        blocks = page_fitz.get_text("dict", flags=fitz.TEXT_DEHYPHENATE)["blocks"]
        font_sizes = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span["size"])
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0

        for block in blocks:
            block_text = ""
            max_font_size = 0
            bold_flags = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span["text"] + " "
                    if span["size"] > max_font_size:
                        max_font_size = span["size"]
                    bold_flags.append("bold" in span["font"].lower())
            block_text = block_text.strip()
            if not block_text:
                continue
            bbox = block.get("bbox", None)
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            is_bold = any(bold_flags)
            is_heading = (
                (max_font_size > avg_font_size * 1.01) and
                (is_bold) and
                (x0 < 100) and
                (len(block_text) < 150)
            )
            blocks_data.append({
                "page": page_num + 1,
                "text": block_text,
                "is_heading": is_heading,
                "is_table": False,
                "font_size": max_font_size,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "bbox": bbox,
            })

        # Second: extract tables if any
        tables = page_plumber.extract_tables()
        for table in tables:
            for row in table:
                for cell in row:
                    if cell and cell.strip():
                        clean_cell = dehyphenate_text(cell.strip())
                        blocks_data.append({
                            "page": page_num + 1,
                            "text": clean_cell,
                            "is_heading": False,
                            "is_table": True,
                            "font_size": None,
                            "x0": None,
                            "y0": None,
                            "x1": x1,
                            "y1": y1,
                            "bbox": None,
                        })
    plumber_pdf.close()
    df_blocks = pd.DataFrame(blocks_data)
    end_time = time.time()
    st.write(f"Time taken for extract_blocks_and_tables: {end_time - start_time:.2f} seconds")
    return df_blocks

def detect_header_footer(df_blocks, y_tolerance=10, min_repeats=50):
    start_time = time.time()
    y_positions = df_blocks['y0'].round(1)
    y_counts = y_positions.value_counts()
    frequent_y = y_counts[y_counts >= min_repeats].index.tolist()
    header_candidates = [y for y in frequent_y if y < 150]
    footer_candidates = [y for y in frequent_y if y > df_blocks['y1'].max() - 150]

    def classify_block(row):
        is_header = any(abs(row['y0'] - hy) <= y_tolerance for hy in header_candidates)
        is_footer = any(abs(row['y0'] - fy) <= y_tolerance for fy in footer_candidates)
        return pd.Series({"is_header": is_header, "is_footer": is_footer})

    header_footer_flags = df_blocks.apply(classify_block, axis=1)
    df_blocks = pd.concat([df_blocks, header_footer_flags], axis=1)
    end_time = time.time()
    st.write(f"Time taken for detect_header_footer: {end_time - start_time:.2f} seconds")
    return df_blocks

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def get_word_count(text):
    return len(re.findall(r'\S+', clean_text(text)))

def enrich_dataframe(df):
    start_time = time.time()
    df['language_detected'] = df['text'].apply(lambda text: detect_major_language_lingua(text))
    df['word_count'] = df['text'].apply(get_word_count)
    end_time = time.time()
    st.write(f"Time taken for enrich_dataframe (Lingua Detection): {end_time - start_time:.2f} seconds")
    return df

detector = LanguageDetectorBuilder.from_all_languages().build()

def clean_for_ngrams(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_ngrams(text, n=3):
    tokens = text.split()
    if len(tokens) < n:
        return tokens
    ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return ngrams

def detect_major_language_lingua(text, n=3):
    text = clean_for_ngrams(text)
    ngrams = generate_ngrams(text, n)
    lang_counter = Counter()
    for ngram in ngrams:
        try:
            lang = detector.detect_language_of(ngram)
            if lang is not None:
                lang_counter[lang] += 1
        except Exception:
            continue
    if lang_counter:
        major_lang = lang_counter.most_common(1)[0][0]
        return major_lang.iso_code_639_1.name.lower()
    else:
        return 'unknown'

def avg_word_count_per_line(text):
    lines = text.split("\n")
    word_counts = [len(line.split()) for line in lines if line.strip()]
    if word_counts:
        return sum(word_counts) / len(word_counts)
    return 0.0

def dictionary_word_percent(text, major_lang):
    if not isinstance(text, str) or not major_lang:
        return 0.0
    words = clean_text(text).split()
    if not words:
        return 0.0
    valid_count = sum(
        1 for word in words
        if word_frequency(word, major_lang, wordlist='best') > 0
    )
    return round((valid_count / len(words)), 2)

def percent_numeric_tokens(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = re.findall(r'\S+', text)
    if not tokens:
        return 0.0
    numeric_pattern = re.compile(r'^([-+]?\d+([.,/]\d+)?)([-–—x×*]\d+([.,/]\d+)?)*$')
    num_count = sum(1 for token in tokens if numeric_pattern.match(token.strip()))
    return num_count / len(tokens)

def is_garbage_line(text, major_lang, symbol_thresh=0.3, dict_thresh=0.3):
    if not isinstance(text, str) or not text.strip():
        return True
    tokens = text.strip().split()
    if not tokens:
        return True
    symbol_count = sum(1 for char in text if char in string.punctuation or char in "[]()/")
    symbol_density = symbol_count / len(text)
    dict_words = sum(1 for token in tokens if word_frequency(token.lower(), major_lang) > 1e-6)
    dict_fraction = dict_words / len(tokens)
    cap_only_tokens = sum(1 for token in tokens if token.isupper() and len(token) > 1)
    cap_token_fraction = cap_only_tokens / len(tokens)
    alpha_num_codes = sum(1 for token in tokens if re.match(r"^[A-Z]{1,5}\d{1,5}$", token))
    code_fraction = alpha_num_codes / len(tokens)
    return (
        symbol_density > symbol_thresh or
        dict_fraction < dict_thresh or
        cap_token_fraction > 0.5 or
        code_fraction > 0.3
    )

def count_single_char_tokens(text):
    if not isinstance(text, str):
        return 0
    tokens = re.findall(r'\S+', text)
    return sum(1 for token in tokens if len(token) == 1) / len(tokens) if tokens else 0

def detect_languages_batch(texts):
    start_time = time.time()
    url = f"https://translation.googleapis.com/language/translate/v2/detect?key={G_API_KEY}"
    data = [('q', text) for text in texts]
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        detections = response.json()['data']['detections']
        languages = [d[0]['language'] for d in detections]
        end_time = time.time()
        st.write(f"Time taken for Google API call: {end_time - start_time:.2f} seconds")
        return languages
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Google Translate API: {e}")
        end_time = time.time()
        st.write(f"Time taken for Google API call (error): {end_time - start_time:.2f} seconds")
        return [None] * len(texts)
    except (KeyError, ValueError) as e:
        st.error(f"Error parsing Google Translate API response: {e}")
        end_time = time.time()
        st.write(f"Time taken for Google API call (error): {end_time - start_time:.2f} seconds")
        return [None] * len(texts)

# --- Streamlit App ---
st.title("PDF Foreign Language Detection")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        pdf_path = "temp.pdf"

        df_blocks = extract_blocks_and_tables(pdf_path)
        # st.write("df_blocks.head():", df_blocks.head().to_string())
        # st.write("df_blocks.shape:", df_blocks.shape)
        if df_blocks.empty:
            st.error("df_blocks is empty after extract_blocks_and_tables.")
            st.stop()  # Stop processing if df_blocks is empty

        df_final = detect_header_footer(df_blocks)
        # st.write("df_final.head():", df_final.head().to_string())
        # st.write("df_final.shape:", df_final.shape)
        if df_final.empty:
            st.error("df_final is empty after detect_header_footer.")
            st.stop()

        df_final_detail = enrich_dataframe(df_final)
        # st.write("df_final_detail.head():", df_final_detail.head().to_string())
        # st.write("df_final_detail['language_detected'].value_counts():", df_final_detail['language_detected'].value_counts())
        # st.write("df_final_detail['word_count'].describe():", df_final_detail['word_count'].describe())
        if df_final_detail.empty:
            st.error("df_final_detail is empty after enrich_dataframe.")
            st.stop()

        df_clean_filter = ((df_final_detail['is_header'] == False) & (df_final_detail['is_footer'] == False) & (df_final_detail['word_count'] >= 3))
        df_clean = df_final_detail.loc[df_clean_filter].copy()
        # st.write("df_clean.head():", df_clean.head().to_string())
        st.write("Total text blocks identified:", df_clean.shape[0])
        if df_clean.empty:
            st.error("df_clean is empty after filtering for non-header/footer and word count >= 3.")
            st.stop()

        df_clean.drop_duplicates(subset='text', keep='first', inplace=True)
        # st.write("df_clean (after duplicates removed).shape:", df_clean.shape)
        # st.write("df_clean['language_detected'].value_counts():", df_clean['language_detected'].value_counts())
        if not df_clean['language_detected'].empty:
            major_lang = df_clean['language_detected'].value_counts().idxmax()
            st.write(f"major language identified is {major_lang}")
            df_foreign_filter = (df_clean['language_detected'] != major_lang)
            df_foreign = df_clean.loc[df_foreign_filter].copy()
            # st.write("df_foreign.head():", df_foreign.head().to_string())
            st.write("Total Foreign text blocks identified:", df_foreign.shape[0])
            if df_foreign.empty:
                st.info(f"df_foreign is empty after filtering for language != major_lang ('{major_lang}').")
            else:
                df_foreign_no_toc_filter = ~df_foreign['text'].str.contains(r'(\.\s*){3,}', regex=True)
                df_foreign = df_foreign.loc[df_foreign_no_toc_filter].copy()
                st.write("Total text blocks after removing table of contents", df_foreign.shape[0])
                if df_foreign.empty:
                    st.info("df_foreign is empty after removing TOC-like lines.")
                else:
                    start_time = time.time()
                    df_foreign.loc[:, 'revised_language_detected'] = df_foreign['text'].apply(lambda x: detect_major_language_lingua(x, n=3))
                    df_foreign.loc[:, 'avg_word_count_per_line'] = df_foreign['text'].apply(avg_word_count_per_line)
                    df_foreign.loc[:, 'percent_numeric_tokens'] = df_foreign['text'].apply(percent_numeric_tokens)
                    df_foreign.loc[:, 'dict_word_percent'] = df_foreign.apply(lambda row: dictionary_word_percent(row['text'], major_lang), axis=1)
                    df_foreign.loc[:, 'is_garbage'] = df_foreign['text'].apply(lambda x: is_garbage_line(x, major_lang=major_lang))
                    df_foreign.loc[:, 'single_char_count'] = df_foreign['text'].apply(count_single_char_tokens)
                    end_time = time.time()
                    st.write(f"Time taken for feature engineering: {end_time - start_time:.2f} seconds")

                    df_foreign_to_google_filter = (
                        (df_foreign['avg_word_count_per_line'] > 2) &
                        (df_foreign['avg_word_count_per_line'] < 25) &
                        (df_foreign['revised_language_detected'] != major_lang) &
                        (df_foreign['dict_word_percent'] > 0.35) &
                        (df_foreign['percent_numeric_tokens'] <= 0.1) &
                        (df_foreign['is_garbage'] == False) &
                        (df_foreign['single_char_count'] <= 0.25)
                    )
                    df_foreign_to_google = df_foreign.loc[df_foreign_to_google_filter].copy()
                    # st.write("df_foreign_to_google.head():", df_foreign_to_google.head().to_string())
                    st.write("Finally, total foreign text blocks being validated via google API ", df_foreign_to_google.shape)
                    if df_foreign_to_google.empty:
                        st.info("df_foreign_to_google is empty after final filtering.")
                    else:
                        df_foreign_to_google_no_toc_filter = ~df_foreign_to_google['text'].str.contains(r'(\.\s*){3,}', regex=True)
                        df_foreign_to_google_no_toc = df_foreign_to_google.loc[df_foreign_to_google_no_toc_filter].copy()
                        # st.write("df_foreign_to_google_no_toc.head():", df_foreign_to_google_no_toc.head().to_string())
                        # st.write("df_foreign_to_google_no_toc.shape:", df_foreign_to_google_no_toc.shape)

                        batch_size = 100
                        results = []
                        google_start_time = time.time() # start timing
                        for i in range(0, len(df_foreign_to_google_no_toc), batch_size):
                            batch = df_foreign_to_google_no_toc['text'].iloc[i:i + batch_size].tolist()
                            langs = detect_languages_batch(batch)
                            results.extend(langs)
                        google_end_time = time.time()
                        st.write(f"Time taken for Google API all batches: {google_end_time - google_start_time:.2f} seconds")
                        df_foreign_to_google_no_toc.loc[:, 'language_google'] = results

                        df_display = df_foreign_to_google_no_toc.loc[df_foreign_to_google_no_toc['language_google'] != major_lang][['page', 'text', 'word_count', 'language_google']].reset_index(drop=True)
                        st.subheader("Detected Foreign Language Blocks")
                        if not df_display.empty:
                            st.dataframe(df_display)

                            @st.cache_data
                            def convert_df_to_csv(df):
                                return df.to_csv(index=False).encode('utf-8')

                            csv_data = convert_df_to_csv(df_display)

                            st.download_button(
                                label="Download detected foreign language blocks as CSV",
                                data=csv_data,
                                file_name="foreign_language_blocks.csv",
                                mime="text/csv",
                            )
                        else:
                            st.info("No foreign language blocks found based on the analysis.")
        else:
            st.info("Could not extract text blocks from the PDF.")
