from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import markdown
import pandas as pd
import json
import csv
import concurrent.futures
import random
import time

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# ====== 전역 변수 ======
user_setups = {}  # 사용자 설정 저장
response_cache = {}  # 응답 캐시

# ====== 상수 ======
GEN_MAX_OUTPUT_TOKENS = 4096
GEN_TIMEOUT_SEC = 180
MD_TIMEOUT_SEC = 10
RETRY_MAX = 3
RETRY_BASE = 2
RETRY_JITTER = 1

# ====== RAG 정책 상수 ======
K_TOTAL = 12  # 후보 문서 총량
K_ANSWER = 6  # 최종 스니펫 수
SNIPPET_MIN_LENGTH = 450
SNIPPET_MAX_LENGTH = 700
DUPLICATE_THRESHOLD = 0.7  # Jaccard 유사도 임계값
RECENCY_BUFFER_DAYS = 90  # 최신성 버퍼 (일)

# RAG 정책 매핑 (라우팅별 쿼터 및 가중치)
RAG_POLICY = {
    "default": {
        "quota": {"card": 6, "bizcsv": 3, "did": 1, "event": 2},
        "w": {"card": 1.0, "bizcsv": 0.85, "did": 0.9, "event": 0.8}
    },
    "trend": {
        "quota": {"card": 4, "bizcsv": 2, "did": 1, "event": 5},
        "w": {"card": 1.0, "bizcsv": 0.85, "did": 0.9, "event": 1.0}
    },
    "retention": {
        "quota": {"card": 5, "bizcsv": 3, "did": 2, "event": 2},
        "w": {"card": 1.0, "bizcsv": 0.85, "did": 1.0, "event": 0.8}
    },
    "diagnosis": {
        "quota": {"card": 5, "bizcsv": 4, "did": 2, "event": 1},
        "w": {"card": 1.0, "bizcsv": 0.9, "did": 0.95, "event": 0.6}
    },
    "loyalty": {
        "quota": {"card": 5, "bizcsv": 4, "did": 1, "event": 2},
        "w": {"card": 1.0, "bizcsv": 0.9, "did": 0.9, "event": 0.8}
    },
    "channel": {
        "quota": {"card": 5, "bizcsv": 4, "did": 1, "event": 2},
        "w": {"card": 1.0, "bizcsv": 0.9, "did": 0.85, "event": 0.85}
    }
}

# 전문가 역할 매핑
EXPERT_ROLES = {
    "trend": "트렌드 분석 전문가",
    "retention": "고객 유지 전문가", 
    "diagnosis": "문제 진단 전문가",
    "loyalty": "고객 충성도 전문가",
    "channel": "마케팅 채널 전문가",
    "default": "성동구 마케팅 전문가"
}

# ====== RAG 유틸리티 함수 ======
def jaccard_similarity(text1, text2):
    """Jaccard 유사도 계산"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0

def simple_bm25_score(text, query_terms):
    """간단한 BM25 스코어 계산"""
    text_lower = text.lower()
    score = 0
    for term in query_terms:
        count = text_lower.count(term.lower())
        if count > 0:
            score += count * 1.2  # 간단한 가중치
    return score

def entity_match_score(text, location="", industry="", task_keywords=None):
    """엔티티 매칭 스코어 계산"""
    score = 0
    text_lower = text.lower()
    
    # 지역/업종 일치
    if location and location.lower() in text_lower:
        score += 0.5
    if industry and industry.lower() in text_lower:
        score += 0.5
    if location and industry and location.lower() in text_lower and industry.lower() in text_lower:
        score += 0.2  # 둘 다 일치 시 추가 보너스
    
    # 태스크 키워드 매칭
    if task_keywords:
        for keyword in task_keywords:
            if keyword.lower() in text_lower:
                score += 0.1
        score = min(score, 0.3)  # 상한 0.3
    
    return score

def recency_score(text, source_type, today=None):
    """최신성 스코어 계산"""
    if not today:
        from datetime import datetime
        today = datetime.now()
    
    score = 0
    if source_type == "event":
        # 이벤트는 날짜 정보 추출하여 최신성 계산
        # 실제 구현에서는 날짜 파싱 로직 필요
        score = 0.5  # 기본값
    else:
        # 다른 소스는 낮은 최신성 영향
        score = 0.1
    
    return score

def numerics_score(text):
    """수치 포함 스코어 계산"""
    import re
    # 퍼센트, 숫자, 시간대 패턴 매칭
    patterns = [
        r'\d+%',  # 퍼센트
        r'\d+\.\d+',  # 소수점
        r'\d+:\d+',  # 시간
        r'\d+-\d+시'  # 시간대
    ]
    
    score = 0
    for pattern in patterns:
        matches = re.findall(pattern, text)
        score += len(matches) * 0.05
    
    return min(score, 0.2)  # 상한 0.2

def diversity_boost_score(text, picked_texts):
    """다양성 부스트 스코어 계산"""
    if not picked_texts:
        return 0
    
    # 동일 파일 연속 과밀 시 감점
    penalty = 0
    for picked in picked_texts[-2:]:  # 최근 2개와 비교
        if jaccard_similarity(text, picked) > 0.8:
            penalty += 0.1
    
    return -penalty

def calculate_final_score(item, source_type, policy_weights, picked_items=None):
    """최종 스코어 계산"""
    text = item.get('content', '')
    location = item.get('location', '')
    industry = item.get('industry', '')
    
    # 각 컴포넌트 스코어 계산
    bm25_score = simple_bm25_score(text, [location, industry]) if location or industry else 0.5
    entity_score = entity_match_score(text, location, industry)
    recency_score_val = recency_score(text, source_type)
    source_weight = policy_weights.get(source_type, 1.0)
    numerics_score_val = numerics_score(text)
    diversity_score = diversity_boost_score(text, [p.get('content', '') for p in picked_items or []])
    
    # 가중 합계
    final_score = (
        0.35 * bm25_score +
        0.20 * entity_score +
        0.15 * recency_score_val +
        0.15 * source_weight +
        0.10 * numerics_score_val +
        0.05 * diversity_score
    )
    
    return final_score

def truncate_with_sentence(text, max_length):
    """문장 경계를 유지하며 텍스트 자르기"""
    if len(text) <= max_length:
        return text
    
    # 문장 단위로 자르기
    sentences = text.split('. ')
    result = ""
    for sentence in sentences:
        if len(result + sentence + '. ') <= max_length:
            result += sentence + '. '
        else:
            break
    
    return result.strip()

def format_citation(source_type, source_file, line_num=None):
    """출처 표기 포맷 생성"""
    citation_map = {
        "card": f"신한카드분석.jsonl#{line_num or 'unknown'}",
        "bizcsv": f"업종별데이터/{source_file}#row{line_num or 'unknown'}",
        "did": f"did.csv#row{line_num or 'unknown'}",
        "event": f"{source_file}#row{line_num or 'unknown'}"
    }
    return f"(출처: {citation_map.get(source_type, source_file)})"

# ====== 유틸리티 함수 ======
def safe_markdown(text):
    """안전한 마크다운 변환"""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            f = ex.submit(markdown.markdown, text, extensions=['extra'])
            try:
                return f.result(timeout=MD_TIMEOUT_SEC)
            except Exception:
                return text
    except Exception:
        return text

def is_overloaded_error(msg):
    """오버로드 오류 감지"""
    overloaded_keywords = ['overloaded', 'quota', 'limit', 'rate']
    return any(keyword in msg.lower() for keyword in overloaded_keywords)

def get_model():
    """모델 가져오기"""
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️ 경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
        return None
    else:
        print("✅ Google API Key가 설정되었습니다.")
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("🤖 Gemini 모델 로드 완료")
        return model

def load_calendar_events():
    """
    달력 이벤트 로드(두 CSV 통합)
    - 컬럼명은 질문에 올려주신 헤더와 동일하게 사용
    - 날짜 'YYYY.M.D' → ISO8601
    - end는 exclusive로 +1일
    - 인코딩 자동 판별
    - 중복 Event_ID 제거
    """
    from datetime import datetime, timedelta
    
    def _read_csv_any(path):
        """UTF-8(BOM) → CP949 → EUC-KR 순으로 시도"""
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                df = pd.read_csv(path, encoding=enc)
                return df
            except Exception:
                continue
        return None

    def _parse_dot_date(s: str) -> str:
        """
        '2023.1.9' → '2023-01-09'
        공백/None/빈값 안전 처리
        """
        if not s or not str(s).strip():
            return ""
        s = str(s).strip().rstrip(".")            # '2023.1.9.' 같은 꼬리 점 제거
        try:
            dt = datetime.strptime(s, "%Y.%m.%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            # 혹시 '2023.01.09' 처럼 0패딩이 있어도 위 포맷이 처리하니 여기로 잘 안옴.
            # 예외 시 원문 반환(디버깅을 위해)
            return s

    def _norm_row(row: dict, src: str, idx: int) -> dict:
        """CSV 한 행을 이벤트 객체로 정규화"""
        start = _parse_dot_date(row.get("Start_Date", ""))
        end_inclusive = _parse_dot_date(row.get("End_Date", ""))

        # FullCalendar 등은 end가 exclusive이므로, 종료일이 있으면 하루 +1
        end_exclusive = ""
        if end_inclusive:
            try:
                end_exclusive = (datetime.strptime(end_inclusive, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            except Exception:
                end_exclusive = end_inclusive  # 실패 시라도 그대로

        eid = row.get("Event_ID") or f"{src}_{idx}"
        title = row.get("Event_Name", "").strip()
        evtype = row.get("Event_Type", "").strip()
        location = row.get("Location_Address", "") or ""
        audience = row.get("Target_Audience", "") or ""
        district = row.get("Associated_District", "") or ""
        desc = row.get("Event_Description", "") or ""

        return {
            "id": str(eid),
            "title": title,
            "start": start,
            "end": end_exclusive,      # ← exclusive로 전달
            "allDay": True,            # 날짜 단위 이벤트 가정
            "type": evtype,
            "location": location,
            "audience": audience,
            "district": district,
            "description": desc,
            "source": src,
        }

    events = []
    seen_ids = set()

    files = [
        ("공통이벤트", os.path.join(app.root_path, "documents", "raw", "성동구 공통_한양대_흥행영화 이벤트 DB.csv")),
        ("성수팝업", os.path.join(app.root_path, "documents", "raw", "성수 팝업 최종.csv")),
    ]

    for label, path in files:
        if not os.path.exists(path):
            continue
        df = _read_csv_any(path)
        if df is None:
            print(f"[캘린더] CSV 인코딩 실패: {path}")
            continue

        # 열 이름 공백 제거/표준화(엑셀 저장 시 공백이 끼는 경우 대비)
        df.columns = [str(c).strip() for c in df.columns]

        for i, row in df.iterrows():
            try:
                ev = _norm_row(row, label, i)
                # start가 비면 스킵
                if not ev["start"]:
                    continue
                # 중복 제거(Event_ID 기준)
                if ev["id"] in seen_ids:
                    continue
                seen_ids.add(ev["id"])
                events.append(ev)
            except Exception as e:
                print(f"[캘린더] 행 파싱 실패 {label}#{i}: {e}")

    print(f"📅 총 {len(events)}개 이벤트 로드 완료")
    return events

def load_shinhan_data():
    """신한카드 데이터 로드"""
    shinhan_data = []
    try:
        shinhan_file = os.path.join(app.root_path, 'documents', 'raw', '신한카드분석.jsonl')
        if os.path.exists(shinhan_file):
            with open(shinhan_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        data['line_num'] = line_num
                        data['source_type'] = 'card'
                        shinhan_data.append(data)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error loading Shinhan data: {e}")
    
    return shinhan_data

def search_card_jsonl(query, location="", industry="", top=18):
    """신한카드 JSONL 데이터 검색"""
    shinhan_data = load_shinhan_data()
    results = []
    
    for item in shinhan_data:
        content = str(item.get('content', ''))
        relevance = 0
        
        # 쿼리 매칭
        if query.lower() in content.lower():
            relevance += 2
        
        # 지역/업종 매칭
        if location and location.lower() in content.lower():
            relevance += 1
        if industry and industry.lower() in content.lower():
            relevance += 1
        
        if relevance > 0:
            results.append({
                'content': content,
                'source_type': 'card',
                'source_file': '신한카드분석.jsonl',
                'line_num': item.get('line_num', 0),
                'location': location,
                'industry': industry,
                'relevance': relevance,
                'original_data': item
            })
    
    # 관련성 순으로 정렬하고 상위 N개 반환
    results.sort(key=lambda x: x['relevance'], reverse=True)
    return results[:top]

def search_biz_csv(query, location="", industry="", top=9):
    """업종/지역 CSV 데이터 검색"""
    results = []
    csv_files = []
    
    # 업종별 CSV 파일들 찾기
    biz_dir = os.path.join(app.root_path, 'documents', 'raw', '업종')
    region_dir = os.path.join(app.root_path, 'documents', 'raw', '지역')
    
    for directory in [biz_dir, region_dir]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.csv'):
                    csv_files.append(os.path.join(directory, filename))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            for row_num, row in df.iterrows():
                # CSV 행을 텍스트로 변환
                content = ' '.join([str(val) for val in row.values if pd.notna(val)])
                
                relevance = 0
                if query.lower() in content.lower():
                    relevance += 2
                if location and location.lower() in content.lower():
                    relevance += 1
                if industry and industry.lower() in content.lower():
                    relevance += 1
                
                if relevance > 0:
                    results.append({
                        'content': content,
                        'source_type': 'bizcsv',
                        'source_file': os.path.basename(csv_file),
                        'line_num': row_num + 1,
                        'location': location,
                        'industry': industry,
                        'relevance': relevance,
                        'original_data': row.to_dict()
                    })
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    results.sort(key=lambda x: x['relevance'], reverse=True)
    return results[:top]

def search_did(query, location="", industry="", top=3):
    """DiD 분석 데이터 검색"""
    results = []
    did_file = os.path.join(app.root_path, 'documents', 'raw', 'did.csv')
    
    if os.path.exists(did_file):
        try:
            df = pd.read_csv(did_file, encoding='utf-8')
            for row_num, row in df.iterrows():
                content = ' '.join([str(val) for val in row.values if pd.notna(val)])
                
                relevance = 0
                if query.lower() in content.lower():
                    relevance += 2
                if location and location.lower() in content.lower():
                    relevance += 1
                if industry and industry.lower() in content.lower():
                    relevance += 1
                
                if relevance > 0:
                    results.append({
                        'content': content,
                        'source_type': 'did',
                        'source_file': 'did.csv',
                        'line_num': row_num + 1,
                        'location': location,
                        'industry': industry,
                        'relevance': relevance,
                        'original_data': row.to_dict()
                    })
        except Exception as e:
            print(f"Error reading did.csv: {e}")
    
    results.sort(key=lambda x: x['relevance'], reverse=True)
    return results[:top]

def search_event_csv(query, location="", top=6, after=None):
    """이벤트 CSV 데이터 검색"""
    results = []
    event_files = [
        '성동구 공통_한양대_흥행영화 이벤트 DB.csv',
        '성수 팝업 최종.csv'
    ]
    
    for filename in event_files:
        file_path = os.path.join(app.root_path, 'documents', 'raw', filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                for row_num, row in df.iterrows():
                    content = ' '.join([str(val) for val in row.values if pd.notna(val)])
                    
                    relevance = 0
                    if query.lower() in content.lower():
                        relevance += 2
                    if location and location.lower() in content.lower():
                        relevance += 1
                    
                    # 최신성 체크 (after 날짜 이후만)
                    if after and 'start_date' in row:
                        try:
                            from datetime import datetime
                            event_date = datetime.strptime(str(row['start_date']), '%Y.%m.%d')
                            if event_date < after:
                                continue
                        except:
                            pass
                    
                    if relevance > 0:
                        results.append({
                            'content': content,
                            'source_type': 'event',
                            'source_file': filename,
                            'line_num': row_num + 1,
                            'location': location,
                            'industry': '',
                            'relevance': relevance,
                            'original_data': row.to_dict()
                        })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
    
    results.sort(key=lambda x: x['relevance'], reverse=True)
    return results[:top]

def retrieve_with_policy(task, query, location="", industry="", today=None):
    """RAG 정책에 따른 문서 검색 및 스코어링"""
    policy = RAG_POLICY.get(task, RAG_POLICY["default"])
    buckets = {"card": [], "bizcsv": [], "did": [], "event": []}
    
    # 1) 소스별 1차 후보 검색
    buckets["card"] = search_card_jsonl(query, location, industry, top=policy["quota"]["card"] * 3)
    buckets["bizcsv"] = search_biz_csv(query, location, industry, top=policy["quota"]["bizcsv"] * 3)
    buckets["did"] = search_did(query, location, industry, top=policy["quota"]["did"] * 3)
    
    # 최신성 필터링을 위한 날짜 계산
    if today is None:
        from datetime import datetime, timedelta
        today = datetime.now()
        after_date = today - timedelta(days=RECENCY_BUFFER_DAYS)
    else:
        after_date = today - timedelta(days=RECENCY_BUFFER_DAYS)
    
    buckets["event"] = search_event_csv(query, location, top=policy["quota"]["event"] * 3, after=after_date)
    
    # 2) 스코어링 + 정렬
    def rank_items(items, source_type):
        return sorted(items, key=lambda x: calculate_final_score(x, source_type, policy["w"]), reverse=True)
    
    for source_type in buckets:
        buckets[source_type] = rank_items(buckets[source_type], source_type)
    
    # 3) 쿼터 컷 + 중복 제거 + 폴백
    picked = []
    picked_texts = []
    
    for source_type in ["card", "bizcsv", "did", "event"]:
        quota = policy["quota"][source_type]
        source_items = buckets[source_type]
        
        for item in source_items[:quota]:
            # 중복 제거 체크
            is_duplicate = False
            for picked_text in picked_texts:
                if jaccard_similarity(item['content'], picked_text) > DUPLICATE_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                picked.append(item)
                picked_texts.append(item['content'])
    
    # 4) 강제 포함 규칙 체크
    picked = ensure_mandatory_sources(picked, task, buckets, policy)
    
    # 5) 스니펫 길이 조정 + 출처 주석 생성
    snippets = []
    for item in picked[:K_ANSWER]:
        content = truncate_with_sentence(item['content'], SNIPPET_MAX_LENGTH)
        citation = format_citation(item['source_type'], item['source_file'], item['line_num'])
        
        snippets.append({
            'content': content,
            'source_type': item['source_type'],
            'source_file': item['source_file'],
            'line_num': item['line_num'],
            'citation': citation,
            'expert_role': EXPERT_ROLES.get(task, EXPERT_ROLES["default"])
        })
    
    return snippets

def ensure_mandatory_sources(picked, task, buckets, policy):
    """필수 소스 포함 규칙 적용"""
    # 신한카드 최소 2개 강제
    card_count = len([p for p in picked if p['source_type'] == 'card'])
    if card_count < 2 and buckets['card']:
        for item in buckets['card']:
            if item not in picked:
                picked.append(item)
                card_count += 1
                if card_count >= 2:
                    break
    
    # 태스크별 필수 소스 체크
    if task == "trend" and len([p for p in picked if p['source_type'] == 'event']) == 0:
        if buckets['event']:
            picked.append(buckets['event'][0])
    
    if task in ["retention", "diagnosis"] and len([p for p in picked if p['source_type'] == 'did']) == 0:
        if buckets['did']:
            picked.append(buckets['did'][0])
    
    return picked

def detect_task_type(query):
    """질문 유형 감지하여 태스크 라우팅"""
    query_lower = query.lower()
    
    # 트렌드 관련 키워드
    trend_keywords = ['트렌드', '인기', '유행', '시즌', '계절', '벚꽃', '캘린더', '이벤트', '행사', '시즌']
    if any(keyword in query_lower for keyword in trend_keywords):
        return "trend"
    
    # 고객 유지 관련 키워드
    retention_keywords = ['재방문', '고객유지', '리텐션', '단골', '충성', '만족도']
    if any(keyword in query_lower for keyword in retention_keywords):
        return "retention"
    
    # 문제 진단 관련 키워드
    diagnosis_keywords = ['문제', '진단', '분석', '원인', '이유', '왜', '어떻게', '해결']
    if any(keyword in query_lower for keyword in diagnosis_keywords):
        return "diagnosis"
    
    # 고객 충성도 관련 키워드
    loyalty_keywords = ['충성도', '브랜드', '애착', '선호도', '만족']
    if any(keyword in query_lower for keyword in loyalty_keywords):
        return "loyalty"
    
    # 채널 관련 키워드
    channel_keywords = ['채널', '마케팅', '광고', '홍보', '소셜미디어', '온라인', '오프라인']
    if any(keyword in query_lower for keyword in channel_keywords):
        return "channel"
    
    # 기본값
    return "default"

def search_relevant_documents(query, location="", industry=""):
    """기존 호환성을 위한 래퍼 함수"""
    return retrieve_with_policy("default", query, location, industry)

def call_gemini_with_retry(model, prompt: str):
    """안정화된 Gemini API 호출"""
    generation_config = {
        "max_output_tokens": GEN_MAX_OUTPUT_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    last_err = None

    for attempt in range(RETRY_MAX):
        try:
            print(f"🔄 Gemini API 호출 시도 {attempt+1}/{RETRY_MAX}")
            print(f"📝 프롬프트 길이: {len(prompt)}자")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    model.generate_content,
                    prompt,
                    generation_config=generation_config,
                )
                resp = fut.result(timeout=GEN_TIMEOUT_SEC)

            print(f"✅ Gemini API 응답 받음: type={type(resp)}")
            
            # 응답 텍스트 추출
            try:
                text_output = resp.text
            except Exception as e:
                print(f"⚠️ 응답 텍스트 추출 실패: {e}")
                text_output = "죄송합니다. 응답 생성 중 문제가 발생했습니다. 질문을 다시 간단히 해주시거나 다른 방식으로 문의해 주세요."
            
            if not text_output or len(text_output.strip()) < 10:
                text_output = "죄송합니다. 응답 생성 중 문제가 발생했습니다. 질문을 다시 간단히 해주시거나 다른 방식으로 문의해 주세요."
                print("⚠️ 빈 응답으로 인한 안전 문구 사용")
            
            print(f"📤 최종 응답 길이: {len(text_output)}자")
            return text_output

        except concurrent.futures.TimeoutError:
            last_err = TimeoutError(f"generate_content timeout ({GEN_TIMEOUT_SEC}s)")
        except Exception as e:
            last_err = e
            msg = str(e)
            if is_overloaded_error(msg) and attempt < RETRY_MAX - 1:
                backoff = (RETRY_BASE ** attempt) + random.uniform(0, RETRY_JITTER)
                print(f"⚠️ 일시 오류(재시도 {attempt+1}/{RETRY_MAX}): {msg} → {backoff:.2f}s 대기")
                time.sleep(backoff)
            else:
                print(f"❌ API Error: {msg}")
                break

    # 최종 실패 시 에러 메시지 반환
    error_msg = f"API 호출 실패: {str(last_err)}"
    print(f"❌ {error_msg}")
    return f"죄송합니다. 응답 생성 중 문제가 발생했습니다. ({error_msg})"

# ====== 라우트 ======
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/setup')
def setup():
    """설정 페이지"""
    return render_template('setup.html')

@app.route('/chat')
def chat():
    """채팅 페이지"""
    return render_template('chat.html')

@app.route('/api/check-setup', methods=['GET'])
def check_setup():
    """설정 확인"""
    setup_exists = 'default' in user_setups and user_setups['default']
    setup_info = user_setups.get('default', {}) if setup_exists else None
    
    return jsonify({
        'setup_exists': setup_exists,
        'is_setup': setup_exists,
        'setup_info': setup_info
    })

@app.route('/api/setup', methods=['POST'])
def save_setup():
    """사용자 설정 저장"""
    try:
        data = request.get_json()
        location = data.get('location', '')
        industry = data.get('industry', '')
        store_name = data.get('store_name', '')
        
        user_setups['default'] = {
            'location': location,
            'industry': industry,
            'store_name': store_name
        }
        
        print(f"📍 지역: {location}, 🏢 업종: {industry}, 🏪 가게: {store_name}")
        
        return jsonify({'success': True, 'message': '설정이 저장되었습니다.'})
    
    except Exception as e:
        print(f"Error saving setup: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-setup', methods=['GET'])
def get_setup():
    """사용자 설정 조회"""
    setup = user_setups.get('default', {})
    return jsonify(setup)

@app.route('/api/calendar-events', methods=['GET'])
def get_calendar_events():
    """달력 이벤트 조회"""
    try:
        events = load_calendar_events()
        return jsonify(events)
    except Exception as e:
        print(f"Error loading calendar events: {e}")
        return jsonify([])

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """채팅 API"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': '메시지를 입력해주세요.'}), 400
        
        # 사용자 설정 가져오기
        setup = user_setups.get('default', {})
        location = setup.get('location', '')
        industry = setup.get('industry', '')
        store_name = setup.get('store_name', '')
        
        # 캐시 확인
        cache_key = f"{user_message}_{location}_{industry}"
        if cache_key in response_cache:
            print("💾 캐시에서 응답 반환")
            return jsonify({
                'message': response_cache[cache_key],
                'session_id': session_id
            })
        
        # Gemini API 호출
        model = get_model()
        if model is None:
            return jsonify({
                'message': '❌ Google API Key가 설정되지 않았습니다.',
                'status': 'error'
            }), 500
        
        # 태스크 라우팅 (질문 유형 감지)
        task = detect_task_type(user_message)
        
        # 새로운 RAG 시스템으로 문서 검색
        relevant_snippets = retrieve_with_policy(task, user_message, location, industry)
        
        # 프롬프트 구성
        context_info = ""
        if location or industry or store_name:
            context_info = f"""
📍 선택된 지역: {location if location else '미선택'}
🏪 선택된 업종: {industry if industry else '미선택'}
🏢 가게명: {store_name if store_name else '미선택'}

위의 지역과 업종 정보를 고려하여 답변해주세요.
"""
        
        # 전문가 역할 식별
        expert_role = EXPERT_ROLES.get(task, EXPERT_ROLES["default"])
        
        # RAG 컨텍스트 추가 (출처 포함)
        rag_context = ""
        if relevant_snippets:
            rag_context = "\n\n[참고 데이터 - 출처 포함]\n"
            for i, snippet in enumerate(relevant_snippets, 1):
                rag_context += f"{i}. {snippet['content']} {snippet['citation']}\n"
        
        prompt = f"""
안녕하세요! 저는 {expert_role}로서 성동구 소상공인 여러분을 도와드립니다.

{context_info}
{rag_context}

질문: {user_message}

위의 지역과 업종 정보를 반드시 고려하여 답변해주세요.
참고 데이터가 있다면 출처와 함께 적극적으로 활용해주세요.
모든 수치나 주장에는 반드시 출처를 표기해주세요.
답변은 실용적이고 구체적으로 작성해주세요.
"""
        
        # Gemini API 호출
        start_time = time.time()
        response_text = call_gemini_with_retry(model, prompt)
        
        # 마크다운 변환
        html_response = safe_markdown(response_text)
        
        # 캐시 저장
        response_cache[cache_key] = html_response
        
        # 처리 시간 로깅
        processing_time = time.time() - start_time
        print(f"⏱️ API 응답 시간: {processing_time:.2f}초")
        
        return jsonify({
            'message': html_response,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Error in chat API: {e}")
        return jsonify({'error': f'오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    """채팅 리셋"""
    global response_cache
    response_cache.clear()
    return jsonify({'success': True, 'message': '채팅이 리셋되었습니다.'})

@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 Flask 서버 시작: 0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)