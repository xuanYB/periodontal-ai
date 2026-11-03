from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# 页面设置
# =========================================================
st.set_page_config(
    page_title="牙周智防",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

LOW_THRESHOLD = 0.32
HIGH_THRESHOLD = 0.42

ROOT = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    ROOT / "RandomForest_final_model_compressed.pkl",
    ROOT / "05_model" / "RandomForest_final_model_compressed.pkl",
]


# =========================================================
# 模型
# =========================================================
@st.cache_resource
def load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return joblib.load(p), p

    raise FileNotFoundError(
        "未找到 RandomForest_final_model_compressed.pkl。"
        "请把模型文件放在 app.py 同一目录。"
    )


try:
    model, model_path = load_model()

except Exception as e:
    st.error("❌ 模型加载失败")
    st.exception(e)
    st.stop()
# =========================================================
# 函数
# =========================================================
def get_level(score):
    if score < LOW_THRESHOLD:
        return "低风险", "🟢", "17.4%"
    elif score < HIGH_THRESHOLD:
        return "中风险", "🟡", "33.9%"
    else:
        return "高风险", "🔴", "49.9%"


def get_factors(row):
    factors = []

    if row["smoking_status"] == 2:
        factors.append("当前吸烟")
    if row["dental_visit"] == 2:
        factors.append("长期未接受牙科检查")
    elif row["dental_visit"] == 1:
        factors.append("牙科复查间隔较长")
    if row["floss_days"] <= 3:
        factors.append("邻面清洁频率较低")
    if row["diabetes_status"] == 2:
        factors.append("糖尿病")
    elif row["diabetes_status"] == 1:
        factors.append("糖代谢异常倾向")
    if row["BMI"] >= 25:
        factors.append("BMI偏高")
    if row["hypertension"] == 1:
        factors.append("高血压")

    return factors


def get_advice(row, level):
    advice = []

    if level == "高风险":
        advice.append("优先接受专业牙周检查，进一步评估牙周袋深度和临床附着丧失。")
    elif level == "中风险":
        advice.append("加强口腔健康管理，并在近期接受牙周健康筛查。")
    else:
        advice.append("保持良好口腔卫生，并进行规律口腔检查。")

    if row["smoking_status"] == 2:
        advice.append("开展戒烟或控烟管理。")

    if row["dental_visit"] == 2:
        advice.append("尽快恢复专业牙科检查。")
    elif row["dental_visit"] == 1:
        advice.append("适当缩短牙科复查间隔。")

    if row["floss_days"] <= 3:
        advice.append("在专业指导下加强牙线或其他邻面清洁。")

    if row["diabetes_status"] >= 1:
        advice.append("加强血糖及代谢健康管理，并关注牙周健康。")

    if row["BMI"] >= 25:
        advice.append("结合饮食、规律运动进行体重管理。")

    if row["hypertension"] == 1:
        advice.append("继续规范血压及慢病管理。")

    return advice


# =========================================================
# 标题
# =========================================================
st.title("🦷 牙周智防")
st.caption("基于可解释机器学习的牙周炎风险分层与主动健康管理原型")

st.info(
    "实时评估模式：修改任意一项信息，右侧结果会自动更新。"
    "本工具用于中重度牙周炎风险筛查与健康管理演示，不用于临床诊断。"
)

# =========================================================
# 主体：左输入 / 右结果
# =========================================================
left, right = st.columns([1.25, 1], gap="large")

with left:
    st.subheader("① 基础健康信息")

    l1, l2 = st.columns(2)

    with l1:
        age = st.number_input(
            "年龄（岁）", min_value=30, max_value=100, value=50, step=1
        )
        sex_text = st.selectbox("性别", ["男", "女"])
        education_text = st.selectbox(
            "教育程度",
            ["低于9年级", "9–11年级", "高中/GED", "部分大学/副学士", "本科及以上"]
        )
        income_ratio = st.number_input(
            "收入贫困比（INDFMPIR）",
            min_value=0.0, max_value=5.0, value=2.0, step=0.1
        )
        floss_days = st.slider(
            "过去7天邻面清洁天数", min_value=0, max_value=7, value=3
        )

    with l2:
        bmi = st.number_input(
            "BMI（kg/m²）", min_value=10.0, max_value=60.0, value=25.0, step=0.1
        )
        waist = st.number_input(
            "腰围（cm）", min_value=40.0, max_value=200.0, value=90.0, step=0.5
        )
        smoking_text = st.selectbox(
            "吸烟状态", ["从不吸烟", "既往吸烟", "当前吸烟"]
        )
        diabetes_text = st.selectbox(
            "糖尿病状态", ["无糖尿病", "糖尿病前期/临界", "糖尿病"]
        )
        hypertension_text = st.selectbox("高血压", ["否", "是"])
        dental_text = st.selectbox(
            "最近一次牙科就诊", ["1年以内", "1–3年", "超过3年或从未"]
        )

# =========================================================
# 编码并实时预测
# =========================================================
sex_map = {"男": 0, "女": 1}
education_map = {
    "低于9年级": 1,
    "9–11年级": 2,
    "高中/GED": 3,
    "部分大学/副学士": 4,
    "本科及以上": 5
}
smoking_map = {"从不吸烟": 0, "既往吸烟": 1, "当前吸烟": 2}
diabetes_map = {"无糖尿病": 0, "糖尿病前期/临界": 1, "糖尿病": 2}
hypertension_map = {"否": 0, "是": 1}
dental_map = {"1年以内": 0, "1–3年": 1, "超过3年或从未": 2}

X_input = pd.DataFrame([{
    "age": float(age),
    "sex": sex_map[sex_text],
    "education": education_map[education_text],
    "income_ratio": float(income_ratio),
    "BMI": float(bmi),
    "waist": float(waist),
    "smoking_status": smoking_map[smoking_text],
    "diabetes_status": diabetes_map[diabetes_text],
    "hypertension": hypertension_map[hypertension_text],
    "dental_visit": dental_map[dental_text],
    "floss_days": float(floss_days)
}])

try:
    score = float(model.predict_proba(X_input)[0, 1])
except Exception as e:
    with right:
        st.error("❌ 模型预测失败")
        st.exception(e)
    st.stop()

level, icon, observed = get_level(score)
row = X_input.iloc[0]
factors = get_factors(row)
advice = get_advice(row, level)

# =========================================================
# 右侧实时结果
# =========================================================
with right:
    st.subheader("② 实时风险评估")

    m1, m2 = st.columns(2)
    m1.metric("模型风险评分", f"{score:.3f}")
    m2.metric("风险等级", f"{icon} {level}")

    st.progress(max(0.0, min(score, 1.0)))

    if level == "高风险":
        st.error("建议优先接受专业牙周检查。")
    elif level == "中风险":
        st.warning("建议近期接受牙周健康筛查。")
    else:
        st.success("建议继续保持良好口腔卫生与规律检查。")

    st.caption(
        f"研究测试集中，该风险层级的中重度牙周炎观察比例为 {observed}。"
        "该数值仅用于风险分层展示，不代表个人精确患病概率。"
    )

    st.markdown("#### ③ 当前可干预/管理因素")
    if factors:
        for item in factors:
            st.markdown(f"- **{item}**")
    else:
        st.write("当前输入信息中未识别出预设的主要可干预因素。")

    st.markdown("#### ④ 主动健康管理建议")
    for i, item in enumerate(advice, 1):
        st.markdown(f"{i}. {item}")

    st.markdown("#### ⑤ 建议管理路径")
    if level == "高风险":
        st.markdown("**专业牙周检查 → 个体因素干预 → 规律复查**")
    elif level == "中风险":
        st.markdown("**强化口腔卫生 → 近期筛查 → 根据结果进一步管理**")
    else:
        st.markdown("**维持健康行为 → 规律检查 → 动态关注风险变化**")

# =========================================================
# 页脚
# =========================================================
st.divider()

f1, f2, f3, f4 = st.columns(4)
f1.metric("主模型", "Random Forest")
f2.metric("测试集 ROC-AUC", "0.700")
f3.metric("筛查阈值", "0.42")
f4.metric("敏感度", "74.8%")

st.caption(
    f"模型文件：{model_path.name} ｜ 低风险界值：0.32 ｜ 高风险界值：0.42 ｜ "
    "科研/竞赛原型，不替代专业牙周诊断。"
)
