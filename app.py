from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import markdown
import pandas as pd
from datetime import datetime, timedelta

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# Gemini API 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini Flash 2.5 모델 설정 (무료 버전)
model = genai.GenerativeModel('gemini-2.5-flash')

# 대화 히스토리 저장
chat_sessions = {}

# 이벤트 데이터 로드 함수
def load_event_data():
    events = {}
    
    try:
        # 성동구 공통 이벤트 데이터 로드
        common_events_df = pd.read_csv('documents/raw/성동구 공통_한양대_흥행영화 이벤트 DB2.csv', encoding='utf-8')
        
        for _, row in common_events_df.iterrows():
            start_date = pd.to_datetime(row['Start_Date'], format='%Y.%m.%d', errors='coerce')
            end_date = pd.to_datetime(row['End_Date'], format='%Y.%m.%d', errors='coerce')
            
            if pd.isna(start_date) or pd.isna(end_date):
                continue
                
            # 이벤트 타입 분류
            event_type = 'general'
            if '팝업' in str(row['Event_Type']):
                event_type = 'popup'
            elif '대학' in str(row['Event_Type']) or '한양대' in str(row['Event_Name']):
                event_type = 'university'
            elif '영화' in str(row['Event_Type']) or '개봉' in str(row['Event_Name']):
                event_type = 'movie'
            
            # 날짜 범위에 따라 이벤트 추가
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str not in events:
                    events[date_str] = []
                
                events[date_str].append({
                    'name': row['Event_Name'],
                    'type': event_type,
                    'startDate': start_date.strftime('%Y-%m-%d'),
                    'endDate': end_date.strftime('%Y-%m-%d'),
                    'location': row['Location_Address'],
                    'target': row['Target_Audience'],
                    'description': row['Event_Description']
                })
                
                current_date += timedelta(days=1)
                
    except Exception as e:
        print(f"공통 이벤트 데이터 로드 실패: {e}")
    
    try:
        # 성수 팝업 데이터 로드
        popup_events_df = pd.read_csv('documents/raw/성수 팝업 최종2.csv', encoding='utf-8')
        
        for _, row in popup_events_df.iterrows():
            start_date = pd.to_datetime(row['Start_Date'], format='%Y.%m.%d', errors='coerce')
            end_date = pd.to_datetime(row['End_Date'], format='%Y.%m.%d', errors='coerce')
            
            if pd.isna(start_date) or pd.isna(end_date):
                continue
            
            # 날짜 범위에 따라 이벤트 추가
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str not in events:
                    events[date_str] = []
                
                events[date_str].append({
                    'name': row['Event_Name'],
                    'type': 'popup',
                    'startDate': start_date.strftime('%Y-%m-%d'),
                    'endDate': end_date.strftime('%Y-%m-%d'),
                    'location': row['Location_Address'],
                    'target': row['Target_Audience'],
                    'description': row['Event_Description']
                })
                
                current_date += timedelta(days=1)
                
    except Exception as e:
        print(f"팝업 이벤트 데이터 로드 실패: {e}")
    
    return events

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calendar')
def calendar():
    return render_template('calendar.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': '메시지를 입력해주세요.'}), 400
        
        # 세션별 채팅 히스토리 관리
        if session_id not in chat_sessions:
            chat_sessions[session_id] = model.start_chat(history=[])
        
        chat = chat_sessions[session_id]
        
        # 성동구 소상공인 마케팅 도우미 시스템 프롬프트
        system_context = """당신은 성동구 지역 소상공인을 위한 전문 마케팅 도우미입니다.

성동구 지역 정보:
- 성수동: 트렌디한 카페, 팝업스토어, 젊은 층 중심
- 왕십리: 전통시장, 중앙시장, 전통과 현대 공존
- 응봉동: 주거지역, 가족 중심
- 옥수동: 한강 근처, 레저업종 유리
- 금호동: 전통 상업지역

당신의 역할:
1. 성동구 지역 특성을 고려한 맞춤형 마케팅 조언
2. 예산별 실용적인 전략 제안
3. SNS, 오프라인, 이벤트 등 다양한 마케팅 방법 안내
4. 마케팅 템플릿과 구체적인 실행 방법 제공

답변 형식:
- 마크다운 형식으로 구조화된 답변
- 제목, 목록, 강조 등을 활용
- 구체적이고 실행 가능한 조언"""
        
        # 첫 메시지에 시스템 컨텍스트 추가
        if len(chat.history) == 0:
            full_message = f"{system_context}\n\n사용자: {user_message}"
        else:
            full_message = user_message
        
        # Gemini에 메시지 전송 (타임아웃 설정)
        import time
        start_time = time.time()
        
        # 간단한 Gemini API 호출
        try:
            direct_model = genai.GenerativeModel('gemini-2.5-flash')
            
            # 간단한 프롬프트로 테스트
            prompt = f"성동구 소상공인 마케팅 도우미입니다. 사용자 질문: {user_message}"
            
            response = direct_model.generate_content(prompt)
            response_text = response.text
            print(f"✅ Gemini API 응답 성공: {response_text[:50]}...")
                
        except Exception as api_error:
            print(f"❌ API Error: {str(api_error)}")
            response_text = f"죄송합니다. 일시적인 오류가 발생했습니다: {str(api_error)}"
        
        # response_text가 이미 설정되어 있음 (테스트 응답 또는 API 응답)
        
        # 마크다운을 HTML로 변환
        html_response = markdown.markdown(response_text, extensions=['extra'])
        
        return jsonify({
            'response': html_response,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/test-gemini', methods=['GET'])
def test_gemini():
    """Gemini API 직접 테스트"""
    try:
        direct_model = genai.GenerativeModel('gemini-2.5-flash')
        response = direct_model.generate_content('안녕하세요, 성동구 소상공인 마케팅 도우미입니다.')
        response_text = response.text
        html_response = markdown.markdown(response_text, extensions=['extra'])
        return jsonify({'response': html_response, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/reset', methods=['POST'])
def reset():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        
        return jsonify({'message': '대화가 초기화되었습니다.'})
    
    except Exception as e:
        return jsonify({'error': f'오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/calendar-events', methods=['GET'])
def get_calendar_events():
    """달력 이벤트 데이터 API"""
    try:
        events = load_event_data()
        return jsonify({'events': events})
    except Exception as e:
        return jsonify({'error': f'이벤트 데이터 로드 실패: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

