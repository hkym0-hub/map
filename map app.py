import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# ---------------------------
# PAGE SETTINGS
# ---------------------------
st.set_page_config(
    page_title="Crime Map Analyzer",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Crime Map Analyzer")
st.write("지도 기반 + 범죄 데이터를 활용한 대시보드")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crime_data.csv")
    return df

df = load_data()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("🔍 Filter Settings")

crime_types = ["전체"] + sorted(df["crime_type"].unique())
selected_type = st.sidebar.selectbox("범죄 유형 선택", crime_types)

address_input = st.sidebar.text_input(
    "주소 검색 (선택)",
    placeholder="예: 서울시 강남구 역삼동"
)

radius = st.sidebar.slider("반경 거리 (m)", 100, 2000, 800)

show_heatmap = st.sidebar.checkbox("히트맵 보기", value=False)

# ---------------------------
# ADDRESS → LAT/LON
# ---------------------------
def geocode_address(address):
    geolocator = Nominatim(user_agent="crime_app")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    return None, None

if address_input:
    lat, lon = geocode_address(address_input)
else:
    # 기본 중심: 서울 시청
    lat, lon = 37.5665, 126.9780

# ---------------------------
# MAP CREATION
# ---------------------------
m = folium.Map(location=[lat, lon], zoom_start=13)

# ---------------------------
# FILTER DATA
# ---------------------------
filtered_df = df.copy()

if selected_type != "전체":
    filtered_df = filtered_df[filtered_df["crime_type"] == selected_type]

# 거리 필터 적용 (주소 검색 사용 시)
from geopy.distance import geodesic

if address_input:
    def is_within_radius(row):
        return geodesic((lat, lon), (row["lat"], row["lon"])).meters <= radius

    filtered_df = filtered_df[filtered_df.apply(is_within_radius, axis=1)]

# ---------------------------
# MARKERS
# ---------------------------
for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        tooltip=f"{row['crime_type']} | {row['date']}",
        color="red",
        fill=True,
        fill_color="red"
    ).add_to(m)

# ---------------------------
# HEATMAP
# ---------------------------
if show_heatmap:
    from folium.plugins import HeatMap
    heat_data = filtered_df[["lat", "lon"]].values.tolist()
    HeatMap(heat_data).add_to(m)

# ---------------------------
# SHOW MAP
# ---------------------------
st.subheader("📍 지도")
st_folium(m, width=900, height=600)

# ---------------------------
# SHOW DATA TABLE
# ---------------------------
st.subheader("📄 데이터 보기")
st.dataframe(filtered_df)
