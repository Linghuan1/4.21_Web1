# -*- coding: utf-8 -*- # 指定编码为 UTF-8
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 页面基础配置 ---
st.set_page_config(
    page_title="盐城二手房智能分析器",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 常量定义：模型和资源文件路径 ---
MARKET_MODEL_PATH = 'market_segment_lgbm_model.joblib'
PRICE_LEVEL_MODEL_PATH = 'price_level_rf_model.joblib'
REGRESSION_MODEL_PATH = 'unit_price_rf_model.joblib'
SCALER_PATH = 'regression_scaler.joblib'
FEATURE_NAMES_PATH = 'feature_names.joblib'
MAPPINGS_PATH = 'mappings.joblib'

# --- 加载资源函数 (使用缓存) ---
@st.cache_resource
def load_resources():
    """加载所有必要的资源文件 (模型, scaler, 特征名, 映射关系)。"""
    resources = {}
    all_files_exist = True
    required_files = [MARKET_MODEL_PATH, PRICE_LEVEL_MODEL_PATH, REGRESSION_MODEL_PATH,
                      SCALER_PATH, FEATURE_NAMES_PATH, MAPPINGS_PATH]
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 未找到。")
            missing_files.append(file_path)
            all_files_exist = False
    if not all_files_exist:
        print(f"错误：缺少文件 {missing_files}。请确保所有 .joblib 文件与 app.py 在同一目录。")
        return None, missing_files
    try:
        resources['market_model'] = joblib.load(MARKET_MODEL_PATH)
        resources['price_level_model'] = joblib.load(PRICE_LEVEL_MODEL_PATH)
        resources['regression_model'] = joblib.load(REGRESSION_MODEL_PATH)
        resources['scaler'] = joblib.load(SCALER_PATH)
        resources['feature_names'] = joblib.load(FEATURE_NAMES_PATH)
        resources['mappings'] = joblib.load(MAPPINGS_PATH)
        print("所有资源加载成功。")
        print("从文件加载的映射关系:", resources['mappings'])
        print("从文件加载的特征名称:", resources['feature_names'])
        return resources, None
    except Exception as e:
        print(f"加载资源时发生错误: {e}")
        return None, [f"加载错误: {e}"]

resources, load_error_info = load_resources()

# --- 辅助函数 ---
def format_mapping_options_for_selectbox(name_to_code_mapping):
    """为 Streamlit Selectbox 准备选项和格式化函数所需的数据。"""
    if not isinstance(name_to_code_mapping, dict):
        print(f"[格式化错误] 输入非字典: {type(name_to_code_mapping)}")
        return {}
    code_to_display_string = {}
    try:
        # Sort by code (assuming codes are numeric or string-numeric)
        sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        for name, code in sorted_items:
             # Ensure code is treated as int for dictionary keys if possible
             try:
                 code_key = int(code)
                 name_str = str(name)
                 code_to_display_string[code_key] = f"{name_str} ({code_key})"
             except (ValueError, TypeError):
                 # Handle cases where code might not be convertible to int (less common for mappings)
                 code_key = str(code)
                 name_str = str(name)
                 code_to_display_string[code_key] = f"{name_str} ({code_key})"

        return code_to_display_string
    except Exception as e: # Catch broader exceptions during sorting/conversion
        print(f"[格式化错误] 转换/排序时出错: {e}")
        # Fallback: create map without sorting if sorting fails
        fallback_map = {}
        for k, v in name_to_code_mapping.items():
             try:
                 code_key = int(v)
                 fallback_map[code_key] = f"{str(k)} ({code_key})"
             except (ValueError, TypeError):
                 code_key = str(v)
                 fallback_map[code_key] = f"{str(k)} ({code_key})"
        return fallback_map

# --- Streamlit 用户界面主要部分 ---
st.title("🏠 盐城二手房智能分析与预测")
st.markdown("""
欢迎使用盐城二手房分析工具！请在左侧边栏输入房产特征，我们将为您提供三个维度的预测：
1.  **市场细分预测**: 判断房产属于低端、中端还是高端市场。
2.  **价格水平预测**: 判断房产单价是否高于其所在区域的平均水平。
3.  **房产均价预测**: 预测房产的每平方米单价（元/㎡）。
""")
st.markdown("---")

# --- 应用启动时资源加载失败或映射缺失的处理 ---
if not resources:
     st.error("❌ **应用程序初始化失败！**")
     if load_error_info:
         st.warning(f"无法加载必要的资源文件。错误详情:")
         for error in load_error_info:
             st.markdown(f"*   `{error}`")
     else:
         st.warning("无法找到一个或多个必需的资源文件。")
     st.markdown(f"""
        请检查以下几点：
        *   确认以下所有 `.joblib` 文件都与 `app.py` 文件在 **同一个** 目录下:
            *   `{MARKET_MODEL_PATH}`
            *   `{PRICE_LEVEL_MODEL_PATH}`
            *   `{REGRESSION_MODEL_PATH}`
            *   `{SCALER_PATH}`
            *   `{FEATURE_NAMES_PATH}`
            *   `{MAPPINGS_PATH}`
        *   确保 `{MAPPINGS_PATH}` 和 `{FEATURE_NAMES_PATH}` 文件内容有效。
        *   检查运行 Streamlit 的终端是否有更详细的错误信息。
     """)
     st.stop()

# --- 如果资源加载成功 ---
mappings = resources['mappings']
feature_names = resources['feature_names']
market_model = resources['market_model']
price_level_model = resources['price_level_model']
regression_model = resources['regression_model']
scaler = resources['scaler']

# --- 检查资源文件内容 ---
required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价']
required_features = ['market', 'price_level', 'regression']
valid_resources = True
missing_or_invalid = []

for key in required_mappings:
    if key not in mappings or not isinstance(mappings.get(key), dict):
        missing_or_invalid.append(f"映射 '{key}' (来自 {MAPPINGS_PATH})")
        valid_resources = False

for key in required_features:
    if key not in feature_names or not isinstance(feature_names.get(key), list):
        missing_or_invalid.append(f"特征列表 '{key}' (来自 {FEATURE_NAMES_PATH})")
        valid_resources = False

if not valid_resources:
    st.error(f"❌ **资源文件内容错误！**")
    st.warning("以下必需的映射或特征列表在资源文件中缺失、无效或格式错误：")
    for item in missing_or_invalid:
        st.markdown(f"*   {item}")
    st.warning(f"请检查 `{MAPPINGS_PATH}` 和 `{FEATURE_NAMES_PATH}` 文件内容。")
    st.stop()

# --- 侧边栏输入控件 ---
st.sidebar.header("🏘️ 房产特征输入")
st.sidebar.subheader("选择项特征")

selectbox_inputs = {}
categorical_feature_keys = ['方位', '楼层', '所属区域', '房龄'] # Keys for selectbox inputs

# Helper function for selectbox creation to avoid repetition
def create_selectbox(label, mapping_key, help_text, key_suffix):
    try:
        options_map = mappings[mapping_key]
        display_map = format_mapping_options_for_selectbox(options_map)
        # Add "None" option at the beginning
        options_codes = [None] + list(display_map.keys())
        # Format function to handle None
        def format_func(x):
            if x is None:
                return "--- 请选择 ---"
            return display_map.get(x, f"未知代码 ({x})")

        # Use index=0 to default to "--- 请选择 ---"
        selectbox_inputs[mapping_key] = st.sidebar.selectbox(
            f"{label}:",
            options=options_codes,
            index=0, # Default to "--- 请选择 ---"
            format_func=format_func,
            key=f"{key_suffix}_select",
            help=help_text
        )
    except Exception as e:
        st.sidebar.error(f"{label} 选项加载错误: {e}")
        selectbox_inputs[mapping_key] = None # Mark as error

create_selectbox("房屋方位", '方位', "选择房屋的主要朝向。如果未知或不适用，请保留'请选择'。", "orientation")
create_selectbox("楼层位置", '楼层', "选择房屋所在的楼层范围（低、中、高）。如果未知或不适用，请保留'请选择'。", "floor_level")
create_selectbox("所属区域", '所属区域', "选择房产所在的盐城市主要区域。如果未知或不适用，请保留'请选择'。", "district")
create_selectbox("房龄范围", '房龄', "选择房屋的建造年限范围。如果未知或不适用，请保留'请选择'。", "age")


st.sidebar.subheader("数值项特征")
numeric_inputs = {}
numeric_inputs['总价(万)'] = st.sidebar.number_input("总价 (万):", min_value=10.0, max_value=1500.0, value=100.0, step=5.0, format="%.1f", key="total_price", help="输入房产的总价，单位万元。")
numeric_inputs['面积(㎡)'] = st.sidebar.number_input("面积 (㎡):", min_value=30.0, max_value=600.0, value=100.0, step=5.0, format="%.1f", key="area_sqm", help="输入房产的建筑面积，单位平方米。")
numeric_inputs['建造时间'] = st.sidebar.number_input("建造时间 (年份):", min_value=1970, max_value=2025, value=2018, step=1, format="%d", key="build_year", help="输入房屋的建造年份。")
numeric_inputs['楼层数'] = st.sidebar.number_input("总楼层数:", min_value=1, max_value=60, value=18, step=1, format="%d", key="floor_num", help="输入楼栋的总楼层数。")
numeric_inputs['室'] = st.sidebar.number_input("室:", min_value=1, max_value=10, value=3, step=1, format="%d", key="rooms", help="输入卧室数量。")
numeric_inputs['厅'] = st.sidebar.number_input("厅:", min_value=0, max_value=5, value=2, step=1, format="%d", key="halls", help="输入客厅/餐厅数量。")
numeric_inputs['卫'] = st.sidebar.number_input("卫:", min_value=0, max_value=5, value=1, step=1, format="%d", key="baths", help="输入卫生间数量。")

# --- 预测触发按钮 ---
st.sidebar.markdown("---")
if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help="点击这里根据输入的特征进行预测分析"):

    # Combine all inputs
    all_inputs = {**selectbox_inputs, **numeric_inputs}
    error_messages = []
    prediction_results = {} # Store results or status for each model

    # --- Define required features for each model ---
    try:
        market_features_needed = feature_names['market']
        price_level_features_needed = feature_names['price_level']
        regression_features_needed = feature_names['regression']
    except KeyError as e:
        st.error(f"加载特征名称时出错: 缺少键 {e}。请检查 `feature_names.joblib` 文件。")
        st.stop() # Stop execution if feature names are missing

    # --- Helper to check for missing categorical inputs for a given model ---
    def check_missing_categoricals(model_features, all_inputs, categorical_keys):
        missing = []
        for feature in model_features:
            if feature in categorical_keys and all_inputs.get(feature) is None:
                missing.append(feature)
        return missing

    # --- 1. 市场细分预测 ---
    market_pred_label = "待处理"
    market_missing_cats = check_missing_categoricals(market_features_needed, all_inputs, categorical_feature_keys)
    if market_missing_cats:
        market_pred_label = f"数据不足，无法判断 (缺少: {', '.join(market_missing_cats)})"
    else:
        try:
            input_data_market = {feat: all_inputs[feat] for feat in market_features_needed if feat in all_inputs}
            # Check if all *needed* features are present (including numerical if any)
            if len(input_data_market) != len(market_features_needed):
                missing_keys = set(market_features_needed) - set(input_data_market.keys())
                raise ValueError(f"市场细分模型缺少输入特征: {missing_keys}") # Should not happen if all inputs are gathered

            input_df_market = pd.DataFrame([input_data_market])[market_features_needed] # Ensure column order
            market_pred_code = market_model.predict(input_df_market)[0]
            market_output_map_inv = {v: k for k, v in mappings.get('市场类别', {}).items()} # Inverse map: code -> name
            market_pred_label = market_output_map_inv.get(int(market_pred_code), f"预测编码无效 ({market_pred_code})")
        except Exception as e:
            msg = f"市场细分模型预测出错: {e}"
            print(msg)
            error_messages.append(msg)
            market_pred_label = "预测失败"
    prediction_results['market'] = market_pred_label

    # --- 2. 价格水平预测 ---
    price_level_pred_label = "待处理"
    price_level_pred_code = -1 # Default code
    price_level_missing_cats = check_missing_categoricals(price_level_features_needed, all_inputs, categorical_feature_keys)
    if price_level_missing_cats:
         price_level_pred_label = f"数据不足，无法判断 (缺少: {', '.join(price_level_missing_cats)})"
    else:
        try:
            input_data_price_level = {feat: all_inputs[feat] for feat in price_level_features_needed if feat in all_inputs}
            if len(input_data_price_level) != len(price_level_features_needed):
                missing_keys = set(price_level_features_needed) - set(input_data_price_level.keys())
                raise ValueError(f"价格水平模型缺少输入特征: {missing_keys}")

            input_df_price_level = pd.DataFrame([input_data_price_level])[price_level_features_needed]
            price_level_pred_code = int(price_level_model.predict(input_df_price_level)[0])
            price_level_output_map_inv = {v: k for k, v in mappings.get('是否高于区域均价', {}).items()} # Inverse map
            price_level_pred_label = price_level_output_map_inv.get(price_level_pred_code, f"预测编码无效 ({price_level_pred_code})")
        except Exception as e:
            msg = f"价格水平模型预测出错: {e}"
            print(msg)
            error_messages.append(msg)
            price_level_pred_label = "预测失败"
    prediction_results['price_level'] = price_level_pred_label
    prediction_results['price_level_code'] = price_level_pred_code # Store code for coloring

    # --- 3. 回归预测 (均价) ---
    unit_price_pred_display = "待处理" # String for display
    unit_price_pred_value = -1 # Numerical value, -1 indicates error or not calculated
    regression_missing_cats = check_missing_categoricals(regression_features_needed, all_inputs, categorical_feature_keys)
    if regression_missing_cats:
         unit_price_pred_display = f"数据不足，无法判断 (缺少: {', '.join(regression_missing_cats)})"
    else:
        try:
            input_data_reg = {feat: all_inputs[feat] for feat in regression_features_needed if feat in all_inputs}
            if len(input_data_reg) != len(regression_features_needed):
                missing_keys = set(regression_features_needed) - set(input_data_reg.keys())
                raise ValueError(f"均价预测模型缺少输入特征: {missing_keys}")

            input_df_reg = pd.DataFrame([input_data_reg])[regression_features_needed] # Ensure column order
            input_df_reg_scaled = scaler.transform(input_df_reg)
            unit_price_pred_raw = regression_model.predict(input_df_reg_scaled)[0]
            unit_price_pred_value = max(0, unit_price_pred_raw) # Ensure non-negative
            unit_price_pred_display = f"{unit_price_pred_value:,.0f}" # Format for display
        except Exception as e:
            msg = f"均价预测模型预测出错: {e}"
            print(msg)
            error_messages.append(msg)
            unit_price_pred_display = "预测失败"
            unit_price_pred_value = -1
    prediction_results['unit_price_display'] = unit_price_pred_display
    prediction_results['unit_price_value'] = unit_price_pred_value

    # --- 结果显示区域 (左对齐，无描述，颜色保留) ---
    st.markdown("---")
    st.subheader("📈 预测结果分析")

    # Define colors
    market_color = "#1f77b4"  # Blue
    price_level_color_high = "#E74C3C" # Red for '高于'
    price_level_color_low = "#2ECC71" # Green for '不高于'
    price_level_color_default = "#ff7f0e" # Orange (default/error)
    unit_price_color = "#2ca02c" # Green
    insufficient_data_color = "#7f7f7f" # Grey for insufficient data
    error_color = "#d62728" # Darker Red for errors

    col1, col2, col3 = st.columns(3)

    with col1: # 市场细分
        st.markdown(f"<h5 style='color: {market_color}; margin-bottom: 5px;'>市场细分</h5>", unsafe_allow_html=True)
        market_result = prediction_results['market']
        if "数据不足" in market_result:
            st.markdown(f"<p style='font-size: 18px; color: {insufficient_data_color}; margin-bottom: 10px;'>{market_result}</p>", unsafe_allow_html=True)
        elif "失败" in market_result or "无效" in market_result:
             st.markdown(f"<p style='font-size: 18px; font-weight: bold; color: {error_color}; margin-bottom: 10px;'>{market_result}</p>", unsafe_allow_html=True)
        else:
             st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: {market_color}; margin-bottom: 10px;'>{market_result}</p>", unsafe_allow_html=True)


    with col2: # 价格水平
        st.markdown(f"<h5 style='color: {price_level_color_default}; margin-bottom: 5px;'>价格水平 (相对区域)</h5>", unsafe_allow_html=True)
        price_level_result = prediction_results['price_level']
        price_level_code = prediction_results['price_level_code']

        if "数据不足" in price_level_result:
             st.markdown(f"<p style='font-size: 18px; color: {insufficient_data_color}; margin-bottom: 10px;'>{price_level_result}</p>", unsafe_allow_html=True)
        elif "失败" in price_level_result or "无效" in price_level_result:
             st.markdown(f"<p style='font-size: 18px; font-weight: bold; color: {error_color}; margin-bottom: 10px;'>{price_level_result}</p>", unsafe_allow_html=True)
        else:
            # Assign color based on prediction code (assuming 1 means '高于', 0 means '不高于')
             display_color = price_level_color_default # Default
             if price_level_code == 1: display_color = price_level_color_high
             elif price_level_code == 0: display_color = price_level_color_low
             st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: {display_color}; margin-bottom: 10px;'>{price_level_result}</p>", unsafe_allow_html=True)

    with col3: # 均价预测
        st.markdown(f"<h5 style='color: {unit_price_color}; margin-bottom: 5px;'>均价预测</h5>", unsafe_allow_html=True)
        unit_price_result = prediction_results['unit_price_display']
        unit_price_value = prediction_results['unit_price_value']

        if "数据不足" in unit_price_result:
            st.markdown(f"<p style='font-size: 18px; color: {insufficient_data_color}; margin-bottom: 10px;'>{unit_price_result}</p>", unsafe_allow_html=True)
        elif "失败" in unit_price_result or unit_price_value == -1: # Check for error state
            st.markdown(f"<p style='font-size: 18px; font-weight: bold; color: {error_color}; margin-bottom: 10px;'>{unit_price_result}</p>", unsafe_allow_html=True)
        else:
            # Display the formatted price with units
            st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: {unit_price_color}; margin-bottom: 10px;'>{unit_price_result} <span style='font-size: small; color: grey;'>元/㎡</span></p>", unsafe_allow_html=True)


    # --- Final Status Message ---
    st.markdown("---")
    if not error_messages:
         # Check if any prediction was hampered by insufficient data
         insufficient_data_count = sum(1 for res in prediction_results.values() if isinstance(res, str) and "数据不足" in res)
         if insufficient_data_count > 0:
              st.info(f"✅ 分析已尝试。部分预测因缺少必要选择项输入而无法完成，请查看上方结果。")
         else:
              st.success("✅ 分析预测完成！请查看上方结果。")
         st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
    else:
         st.warning("⚠️ 分析过程中遇到错误，部分或全部预测未能完成。")
         for msg in error_messages:
              st.error(f"错误详情: {msg}")

# --- 页脚信息 ---
st.sidebar.markdown("---")
st.sidebar.caption("模型信息: LightGBM & RandomForest")
st.sidebar.caption("数据来源: 安居客")
st.sidebar.caption("开发者: 凌欢")