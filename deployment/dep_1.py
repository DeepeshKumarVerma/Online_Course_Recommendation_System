# FINAL Hybrid Course Recommender (User-ID CF + Input CBF)
import streamlit as st
import pandas as pd
import numpy as np
import pickle, zipfile, os, sys, types
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

st.set_page_config(page_title="Hybrid Course Recommender", layout="wide")

# 1. LOAD MODEL SAFELY (Handle GPU Pickle)
@st.cache_resource
def load_model_bundle(
    zip_path=r"\hybrid_recommender_bundle_1.zip",
    extract_dir="deployed_model_final",
):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # --- Mock GPU modules properly ---
    class MockMatrix:
        def __init__(self, arr=None): self.data = arr
        def __array__(self, dtype=None): return self.data

    class MockALS:
        def __init__(self, *args, **kwargs): pass

    mock_gpu = types.ModuleType("implicit.gpu")
    mock_gpu_als = types.ModuleType("implicit.gpu.als")
    mock_gpu_mfb = types.ModuleType("implicit.gpu.matrix_factorization_base")

    mock_gpu.Matrix = MockMatrix
    mock_gpu.AlternatingLeastSquares = MockALS
    mock_gpu_als.Matrix = MockMatrix
    mock_gpu_als.AlternatingLeastSquares = MockALS
    mock_gpu_mfb.Matrix = MockMatrix
    mock_gpu_mfb.AlternatingLeastSquares = MockALS

    sys.modules["implicit.gpu"] = mock_gpu
    sys.modules["implicit.gpu.als"] = mock_gpu_als
    sys.modules["implicit.gpu.matrix_factorization_base"] = mock_gpu_mfb

    # --- Load model_cf.pkl safely ---
    gpu_model = pickle.load(open(os.path.join(extract_dir, "model_cf.pkl"), "rb"))

    model_cf = AlternatingLeastSquares(
        factors=getattr(gpu_model, "factors", 64),
        regularization=getattr(gpu_model, "regularization", 0.1),
        iterations=getattr(gpu_model, "iterations", 20),
    )

    # ✅ Force pure NumPy arrays to avoid view errors
    model_cf.user_factors = np.asarray(
        getattr(gpu_model.user_factors, "data", gpu_model.user_factors)
    ).copy()
    model_cf.item_factors = np.asarray(
        getattr(gpu_model.item_factors, "data", gpu_model.item_factors)
    ).copy()

    # --- Load supporting assets ---
    def pkl(name):
        return pickle.load(open(os.path.join(extract_dir, f"{name}.pkl"), "rb"))

    feature_matrix = np.asarray(pkl("feature_matrix"))
    course_features = pkl("course_features")
    user_map = pkl("user_map")
    item_inv = pkl("item_inv")
    interaction_matrix = pkl("interaction_matrix")
    if not isinstance(interaction_matrix, csr_matrix):
        interaction_matrix = csr_matrix(interaction_matrix)

    w = pkl("best_weights")
    w_cf, w_cbf = w.get("w_cf", 0.6), w.get("w_cbf", 0.4)

    # optional components
    def opt_pkl(name):
        path = os.path.join(extract_dir, f"{name}.pkl")
        return pickle.load(open(path, "rb")) if os.path.exists(path) else None

    le_course = opt_pkl("le_course")
    le_instructor = opt_pkl("le_instructor")
    le_difficulty = opt_pkl("le_difficulty")
    scaler = opt_pkl("minmax_scaler")

    return {
        "model_cf": model_cf,
        "feature_matrix": feature_matrix,
        "course_features": course_features,
        "user_map": user_map,
        "item_inv": item_inv,
        "interaction_matrix": interaction_matrix,
        "w_cf": w_cf,
        "w_cbf": w_cbf,
        "le_course": le_course,
        "le_instructor": le_instructor,
        "le_difficulty": le_difficulty,
        "scaler": scaler,
    }

assets = load_model_bundle()

# 2. SAFE INPUT PREPROCESSING
def preprocess_user_input(raw_input):
    le_instructor = assets["le_instructor"]
    le_difficulty = assets["le_difficulty"]
    scaler = assets["scaler"]

    processed = {}

    # Encode categorical
    processed["instructor_enc"] = (
        int(le_instructor.transform([raw_input["instructor"]])[0])
        if le_instructor and raw_input["instructor"] in le_instructor.classes_
        else 0
    )
    processed["difficulty_level_enc"] = (
        int(le_difficulty.transform([raw_input["difficulty_level"]])[0])
        if le_difficulty and raw_input["difficulty_level"] in le_difficulty.classes_
        else 0
    )

    # Scale numeric safely
    numeric = np.array([[raw_input["enrollment_numbers"],
                         raw_input["course_price"],
                         raw_input["course_duration_hours"],
                         raw_input["feedback_score"],
                         raw_input["time_spent_hours"],
                         0.0, 0.0]])
    if scaler:
        scaled = scaler.transform(numeric)[0]
    else:
        scaled = numeric[0]
    processed.update({
        "enrollment_numbers": float(scaled[0]),
        "course_price": float(scaled[1]),
        "course_duration_hours": float(scaled[2]),
        "feedback_score": float(scaled[3]),
        "time_spent_hours": float(scaled[4]),
    })
    return processed

# 3. HYBRID RECOMMENDER (No Sparse Sub-Views)
def recommend_hybrid(user_id, input_features, top_n=10):
    model_cf = assets["model_cf"]
    feature_matrix = assets["feature_matrix"]
    course_features = assets["course_features"]
    user_map = assets["user_map"]
    item_inv = assets["item_inv"]
    le_course = assets["le_course"]
    w_cf, w_cbf = assets["w_cf"], assets["w_cbf"]

    # --- CF Personalized Scores (safe dot product) ---
    if user_id not in user_map:
        return f"⚠️ User ID {user_id} not found in training data."
    uid = user_map[user_id]

    # direct dot product instead of model.recommend()
    cf_scores = np.dot(model_cf.item_factors, model_cf.user_factors[uid])
    df_cf = pd.DataFrame({
        "course_name_enc": list(item_inv.values()),
        "cf_score": cf_scores
    })
    df_cf["cf_score_norm"] = df_cf["cf_score"].rank(ascending=False, pct=True)

    # --- CBF ---
    user_vec = np.array([[input_features["instructor_enc"],
                          input_features["difficulty_level_enc"],
                          input_features["enrollment_numbers"],
                          input_features["course_price"],
                          input_features["course_duration_hours"],
                          input_features["feedback_score"],
                          input_features["time_spent_hours"]]])
    sims = cosine_similarity(user_vec, feature_matrix)[0]
    df_cbf = pd.DataFrame({
        "course_name_enc": course_features["course_name_enc"].values,
        "cbf_score": sims
    })
    df_cbf["cbf_score_norm"] = df_cbf["cbf_score"].rank(ascending=False, pct=True)

    # --- Combine CF + CBF ---
    df = pd.merge(df_cf, df_cbf, on="course_name_enc", how="outer").fillna(0)
    df["hybrid_score"] = w_cf * df["cf_score_norm"] + w_cbf * df["cbf_score_norm"]
    df = df.sort_values("hybrid_score", ascending=False).head(top_n)

    # --- Decode course names ---
    if le_course is not None:
        try:
            df["course_name"] = le_course.inverse_transform(df["course_name_enc"].astype(int))
        except Exception:
            lookup = course_features[["course_name_enc", "course_name"]].drop_duplicates()
            df = df.merge(lookup, on="course_name_enc", how="left")
    else:
        if "course_name" in course_features.columns:
            lookup = course_features[["course_name_enc", "course_name"]].drop_duplicates()
            df = df.merge(lookup, on="course_name_enc", how="left")

    # ✅ Drop course_name_enc from final output
    df_final = df[["course_name", "cf_score", "cbf_score", "hybrid_score"]].reset_index(drop=True)
    return df_final

# 4. STREAMLIT UI
st.title("🎓 Course Recommendation System")
st.markdown("CF uses your User ID (from training data), and CBF uses your current preferences.")

user_id = st.number_input("Enter User ID", min_value=1, step=1, value=15796)

instructors = list(getattr(assets["le_instructor"], "classes_", [])) or ["Unknown"]
difficulties = list(getattr(assets["le_difficulty"], "classes_", [])) or ["Beginner"]

with st.form("recommendation_form"):
    st.subheader("🔧 Input Your Preferences")
    instructor = st.selectbox("Instructor", instructors)
    difficulty_level = st.selectbox("Difficulty Level", difficulties)
    enrollment_numbers = st.number_input("Enrollment Numbers", min_value=0, value=500)
    course_price = st.number_input("Course Price (₹)", min_value=0, value=2500)
    course_duration_hours = st.number_input("Course Duration (hours)", min_value=1, value=40)
    
    # ✅ Feedback now 0 to 1
    feedback_score = st.slider("Feedback Score (0–1)", 0.0, 1.0, 0.8)
    
    time_spent_hours = st.number_input("Time Spent (hours)", min_value=0.0, value=10.0)
    top_n = st.slider("Top N Recommendations", 5, 20, 10)
    submitted = st.form_submit_button("🎯 Get Recommendations")

if submitted:
    raw = {
        "instructor": instructor,
        "difficulty_level": difficulty_level,
        "enrollment_numbers": enrollment_numbers,
        "course_price": course_price,
        "course_duration_hours": course_duration_hours,
        "feedback_score": feedback_score,
        "time_spent_hours": time_spent_hours,
    }
    try:
        features = preprocess_user_input(raw)
        recs = recommend_hybrid(int(user_id), features, top_n=top_n)
        st.success("✅ Recommendations generated successfully!")
        st.dataframe(recs)
    except Exception as e:
        st.error(f"❌ Error: {e}")
