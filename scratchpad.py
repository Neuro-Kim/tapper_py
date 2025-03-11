import streamlit as st


# 텍스트 입력
name = st.text_input("이름을 입력하세요")
st.write(f"안녕하세요, {name}님!")

# 숫자 입력
age = st.number_input("나이를 입력하세요", min_value=0, max_value=120, value=25)
st.write(f"당신의 나이는 {age}세입니다.")

# 슬라이더
value = st.slider("값 선택", 0, 100, 50)

# 선택 박스
option = st.selectbox("옵션을 선택하세요", ["옵션1", "옵션2", "옵션3"])

# 다중 선택
options = st.multiselect("여러 옵션을 선택하세요", ["옵션1", "옵션2", "옵션3", "옵션4"])

# 체크박스
if st.checkbox("동의합니다"):
    st.write("감사합니다!")

# 버튼
if st.button("클릭"):
    st.write("버튼이 클릭되었습니다!")

# 날짜 입력
date = st.date_input("날짜 선택")

# 시간 입력
time = st.time_input("시간 선택")

# 파일 업로더
uploaded_file = st.file_uploader("파일 업로드", type=["csv", "txt"])
if uploaded_file is not None:
    st.write("파일이 업로드되었습니다!")
    
    
# 사이드바
with st.sidebar:
    st.title("사이드바 제목")
    name = st.text_input("이름")
    age = st.slider("나이", 0, 100, 25)

# 컬럼 레이아웃
col1, col2, col3 = st.columns(3)
with col1:
    st.header("컬럼 1")
    st.image("image1.jpg")
with col2:
    st.header("컬럼 2")
    st.image("image2.jpg")
with col3:
    st.header("컬럼 3")
    st.image("image3.jpg")

# 탭 레이아웃
tab1, tab2, tab3 = st.tabs(["탭1", "탭2", "탭3"])
with tab1:
    st.header("탭 1 내용")
with tab2:
    st.header("탭 2 내용")
with tab3:
    st.header("탭 3 내용")

# 확장 가능한 컨테이너
with st.expander("더 보기"):
    st.write("숨겨진 내용입니다.")