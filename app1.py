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

# --- 常量定义 ---
MARKET_MODEL_PATH = 'market_segment_lgbm_model.joblib'
PRICE_LEVEL_MODEL_PATH = 'price_level_rf_model.joblib'
REGRESSION_MODEL_PATH = 'unit_price_rf_model.joblib'
SCALER_PATH = 'regression_scaler.joblib'
FEATURE_NAMES_PATH = 'feature_names.joblib'
MAPPINGS_PATH = 'mappings.joblib'

# --- 加载资源函数 (使用缓存) ---
@st.cache_resource
def load_resources():
    """加载所有必要的资源文件。"""
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
        print(f"错误：缺少文件 {missing_files}。")
        return None, missing_files
    try:
        resources['market_model'] = joblib.load(MARKET_MODEL_PATH)
        resources['price_level_model'] = joblib.load(PRICE_LEVEL_MODEL_PATH)
        resources['regression_model'] = joblib.load(REGRESSION_MODEL_PATH)
        resources['scaler'] = joblib.load(SCALER_PATH)
        resources['feature_names'] = joblib.load(FEATURE_NAMES_PATH)
        resources['mappings'] = joblib.load(MAPPINGS_PATH)
        print("所有资源加载成功。")
        return resources, None
    except Exception as e:
        print(f"加载资源时发生错误: {e}")
        return None, [f"加载错误: {e}"]

# --- 辅助函数：格式化下拉框选项 ---
def format_mapping_options_for_selectbox(name_to_code_mapping):
    """为 Streamlit Selectbox 准备选项和格式化函数所需的数据。"""
    if not isinstance(name_to_code_mapping, dict): return {}
    code_to_display_string = {}
    try:
        try:
            sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        except ValueError:
             sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: str(item[1]))
        for name, code in sorted_items:
            try: code_key = int(code)
            except ValueError: code_key = str(code)
            code_to_display_string[code_key] = f"{str(name)} ({code})"
        return code_to_display_string
    except Exception as e:
        print(f"[格式化错误] 格式化选项时出错: {e}")
        return {v: f"{k} ({v})" for k, v in name_to_code_mapping.items()} # Fallback

# --- 加载资源 ---
resources, load_error_info = load_resources()

# --- Streamlit 用户界面 ---
st.title("🏠 盐城二手房智能分析与预测")
st.markdown("""
在左侧输入房产特征，点击按钮开始分析。系统会根据您提供的信息，尝试进行以下预测：
1.  **市场细分**: 房产在市场中的定位。
2.  **价格水平**: 房产单价与其区域均值的比较。
3.  **均价预测**: 房产的每平方米单价。
""")
st.markdown("---")

# --- 资源加载失败处理 ---
if not resources:
     st.error("❌ **应用程序初始化失败！**")
     if load_error_info:
         st.warning("无法加载必要的资源文件。错误详情:")
         for error in load_error_info: st.markdown(f"*   `{error}`")
     else:
         st.warning("无法找到一个或多个必需的资源文件。")
     st.markdown("请检查所有 `.joblib` 文件是否与 `app.py` 在同一目录且文件有效。")
     st.stop()

# --- 资源检查 ---
mappings = resources['mappings']
feature_names = resources['feature_names']
market_model = resources['market_model']
price_level_model = resources['price_level_model']
regression_model = resources['regression_model']
scaler = resources['scaler']

required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价']
required_features = ['market', 'price_level', 'regression']
valid_resources = True
missing_or_invalid = []
for key in required_mappings:
    if key not in mappings or not isinstance(mappings.get(key), dict):
        missing_or_invalid.append(f"映射 '{key}'")
        valid_resources = False
for key in required_features:
    if key not in feature_names or not isinstance(feature_names.get(key), list):
        missing_or_invalid.append(f"特征列表 '{key}'")
        valid_resources = False
if not valid_resources:
    st.error(f"❌ 资源文件内容不完整或格式错误。缺少或无效的项目: {', '.join(missing_or_invalid)}")
    st.stop()

# --- 侧边栏输入控件 ---
st.sidebar.header("🏘️ 房产特征输入")
st.sidebar.info("请尽量提供完整信息以获得更全面的预测。") # 提示用户输入完整性

st.sidebar.subheader("选择项特征")
selectbox_inputs = {}
all_select_valid = True

def create_selectbox(label, mapping_key, help_text, key_suffix):
    global all_select_valid
    try:
        options_map = mappings[mapping_key]
        display_map = format_mapping_options_for_selectbox(options_map)
        if not display_map: raise ValueError(f"映射 '{mapping_key}' 格式化后为空。")
        options_codes = list(display_map.keys())
        default_index = 0
        if options_codes:
            common_defaults = {'楼层': 1, '房龄': 2}
            if mapping_key in common_defaults and common_defaults[mapping_key] in options_codes:
                 try: default_index = options_codes.index(common_defaults[mapping_key])
                 except ValueError: pass
            elif len(options_codes) > 1: default_index = len(options_codes) // 2
        selected_value = st.sidebar.selectbox(label, options=options_codes, index=default_index,
                                            format_func=lambda x: display_map.get(x, f"未知 ({x})"),
                                            key=f"{key_suffix}_select", help=help_text)
        return selected_value
    except Exception as e:
        st.sidebar.error(f"加载 '{label}' 选项出错: {e}")
        all_select_valid = False; return None

selectbox_inputs['方位'] = create_selectbox("房屋方位:", '方位', "选择房屋的主要朝向。", "orientation")
selectbox_inputs['楼层'] = create_selectbox("楼层位置:", '楼层', "选择房屋所在楼层的大致位置（低、中、高）。", "floor_level")
selectbox_inputs['所属区域'] = create_selectbox("所属区域:", '所属区域', "选择房产所在的行政区域或板块。", "district")
selectbox_inputs['房龄'] = create_selectbox("房龄:", '房龄', "选择房屋的建造年限范围。", "age")

st.sidebar.subheader("数值项特征")
numeric_inputs = {}
# --- 创建数值输入，允许用户不输入（或输入特定值表示缺失，但streamlit number_input强制要求有值，所以我们后面判断） ---
# 注意：Streamlit number_input 不直接支持"空"值。如果用户想表示缺失，他们可以保留默认值，
# 或者我们可以在代码逻辑中判断某个值（如0或-1，如果业务允许）代表缺失。
# 这里我们假设用户会输入实际值，如果模型需要的值没输入（或保留了不适用的默认值），模型输入检查会捕捉到。
numeric_inputs['总价(万)'] = st.sidebar.number_input("总价 (万):", min_value=0.0, max_value=3000.0, value=120.0, step=5.0, format="%.1f", key="total_price", help="输入房产的总价（万元）。如果未知，保留默认或输入0（模型会判断是否需要）。")
numeric_inputs['面积(㎡)'] = st.sidebar.number_input("面积 (㎡):", min_value=10.0, max_value=1000.0, value=95.0, step=1.0, format="%.1f", key="area_sqm", help="输入房产的建筑面积（平方米）。")
numeric_inputs['建造时间'] = st.sidebar.number_input("建造时间 (年份):", min_value=1950, max_value=2025, value=2015, step=1, format="%d", key="build_year", help="输入房屋的建造年份。")
numeric_inputs['楼层数'] = st.sidebar.number_input("总楼层数:", min_value=1, max_value=80, value=18, step=1, format="%d", key="floor_num", help="输入楼栋的总楼层数。")
numeric_inputs['室'] = st.sidebar.number_input("室:", min_value=1, max_value=15, value=3, step=1, format="%d", key="rooms", help="输入卧室数量。")
numeric_inputs['厅'] = st.sidebar.number_input("厅:", min_value=0, max_value=10, value=2, step=1, format="%d", key="halls", help="输入客厅/餐厅数量。")
numeric_inputs['卫'] = st.sidebar.number_input("卫:", min_value=0, max_value=8, value=1, step=1, format="%d", key="baths", help="输入卫生间数量。")

# --- 预测触发按钮 ---
st.sidebar.markdown("---")
predict_button_disabled = not all_select_valid
predict_button_help = "点击开始分析预测" if all_select_valid else "部分下拉选项加载失败，无法预测。"

if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help=predict_button_help, disabled=predict_button_disabled):

    # --- 初始化预测结果状态 ---
    market_pred_status = "not_run" # 状态: not_run, success, insufficient_data, error
    market_pred_result = None
    market_missing_features = []
    market_error_msg = ""

    price_level_pred_status = "not_run"
    price_level_pred_result = None # 存储 (标签, 编码)
    price_level_missing_features = []
    price_level_error_msg = ""

    regression_pred_status = "not_run"
    regression_pred_result = None # 存储预测值
    regression_missing_features = []
    regression_error_msg = ""

    runtime_errors = [] # 存储预测执行期间的非输入性错误

    # 合并所有输入
    all_inputs = {**selectbox_inputs, **numeric_inputs}
    if None in selectbox_inputs.values(): # 再次检查下拉框
        st.error("⚠️ 输入错误：检测到无效的下拉选择项。")
    else:
        # --- 1. 尝试市场细分预测 ---
        try:
            market_features_needed = feature_names.get('market', [])
            input_data_market = {}
            market_missing_features = []
            for feat in market_features_needed:
                if feat in all_inputs:
                    input_data_market[feat] = all_inputs[feat]
                else:
                    market_missing_features.append(feat)

            if market_missing_features: # 如果缺少必需特征
                market_pred_status = "insufficient_data"
            else: # 特征齐全，尝试预测
                input_df_market = pd.DataFrame([input_data_market])[market_features_needed]
                market_pred_code = market_model.predict(input_df_market)[0]
                market_output_map = mappings.get('市场类别', {})
                market_pred_key = int(market_pred_code) if isinstance(market_pred_code, (int, np.integer)) else str(market_pred_code)
                market_pred_result = market_output_map.get(market_pred_key, f"未知编码({market_pred_key})")
                market_pred_status = "success"
        except Exception as e:
            market_pred_status = "error"
            market_error_msg = f"市场细分模型运行时出错: {e}"
            runtime_errors.append(market_error_msg)
            print(market_error_msg)

        # --- 2. 尝试价格水平预测 ---
        try:
            price_level_features_needed = feature_names.get('price_level', [])
            input_data_price_level = {}
            price_level_missing_features = []
            for feat in price_level_features_needed:
                if feat in all_inputs:
                    # 特别检查：如果总价是必需的，但用户输入了0或不合理的值，可能也视为不足
                    # 这里简化处理：仅检查是否存在于 all_inputs
                    input_data_price_level[feat] = all_inputs[feat]
                else:
                    price_level_missing_features.append(feat)

            if price_level_missing_features: # 缺少特征
                price_level_pred_status = "insufficient_data"
            else: # 特征齐全
                input_df_price_level = pd.DataFrame([input_data_price_level])[price_level_features_needed]
                price_level_pred_code = price_level_model.predict(input_df_price_level)[0]
                price_level_output_map = mappings.get('是否高于区域均价', {})
                price_level_pred_key = int(price_level_pred_code) if isinstance(price_level_pred_code, (int, np.integer)) else str(price_level_pred_code)
                price_level_label = price_level_output_map.get(price_level_pred_key, f"未知编码({price_level_pred_key})")
                # 存储标签和编码
                price_level_pred_result = (price_level_label, int(price_level_pred_code) if isinstance(price_level_pred_code, (int, np.integer)) else -99)
                price_level_pred_status = "success"
        except Exception as e:
            price_level_pred_status = "error"
            price_level_error_msg = f"价格水平模型运行时出错: {e}"
            runtime_errors.append(price_level_error_msg)
            print(price_level_error_msg)


        # --- 3. 尝试均价预测 ---
        try:
            regression_features_needed = feature_names.get('regression', [])
            input_data_reg = {}
            regression_missing_features = []
            for feat in regression_features_needed:
                 if feat in all_inputs:
                    input_data_reg[feat] = all_inputs[feat]
                 else:
                    regression_missing_features.append(feat)

            if regression_missing_features: # 缺少特征
                regression_pred_status = "insufficient_data"
            else: # 特征齐全
                input_df_reg = pd.DataFrame([input_data_reg])[regression_features_needed]
                input_df_reg_scaled = scaler.transform(input_df_reg)
                unit_price_pred = regression_model.predict(input_df_reg_scaled)[0]
                regression_pred_result = max(0, float(unit_price_pred)) # 存储预测值
                regression_pred_status = "success"
        except Exception as e:
            regression_pred_status = "error"
            regression_error_msg = f"均价预测模型运行时出错: {e}"
            runtime_errors.append(regression_error_msg)
            print(regression_error_msg)


        # --- 结果显示区域 (已简化) ---
        st.markdown("---")
        st.subheader("📈 预测结果分析")

        market_color = "#1f77b4"
        price_level_base_color = "#ff7f0e"
        unit_price_color = "#2ca02c"
        error_color = "#E74C3C"
        success_color = "#2ECC71"
        grey_color = "#7f7f7f"
        warning_color = "#F39C12" # 用于数据不足

        col1, col2, col3 = st.columns(3)

        with col1: # 市场细分
            st.markdown(f"<h5 style='color: {market_color}; margin-bottom: 5px;'>市场细分</h5>", unsafe_allow_html=True)
            if market_pred_status == "success":
                st.markdown(f"<p style='font-size: 28px; font-weight: bold; color: {market_color}; margin-bottom: 10px;'>{market_pred_result}</p>", unsafe_allow_html=True)
            elif market_pred_status == "insufficient_data":
                st.warning("数据不足") # 显示数据不足
            elif market_pred_status == "error":
                st.error("预测出错") # 显示运行时错误
            else: # not_run
                 st.markdown(f"<p style='font-size: small; color: {grey_color};'>未运行</p>", unsafe_allow_html=True)

            with st.expander("查看使用/缺失特征"):
                if market_pred_status == "insufficient_data":
                    st.caption(f"需要但缺失: {', '.join(market_missing_features)}")
                elif market_features_needed:
                    st.caption(f"模型使用: {', '.join(market_features_needed)}")
                if market_pred_status == "error":
                     st.caption(f"错误: {market_error_msg}")


        with col2: # 价格水平
            st.markdown(f"<h5 style='color: {price_level_base_color}; margin-bottom: 5px;'>价格水平 (相对区域)</h5>", unsafe_allow_html=True)
            if price_level_pred_status == "success":
                label, code = price_level_pred_result
                if code == 1: display_color = error_color   # 高于 (红)
                elif code == 0: display_color = success_color # 不高于 (绿)
                else: display_color = grey_color             # 未知 (灰)
                st.markdown(f"<p style='font-size: 28px; font-weight: bold; color: {display_color}; margin-bottom: 10px;'>{label}</p>", unsafe_allow_html=True)
            elif price_level_pred_status == "insufficient_data":
                st.warning("数据不足") # <--- 修改点：显示数据不足
            elif price_level_pred_status == "error":
                st.error("预测出错")
            else: # not_run
                 st.markdown(f"<p style='font-size: small; color: {grey_color};'>未运行</p>", unsafe_allow_html=True)

            with st.expander("查看使用/缺失特征"):
                 if price_level_pred_status == "insufficient_data":
                    st.caption(f"需要但缺失: {', '.join(price_level_missing_features)}")
                 elif feature_names.get('price_level'):
                    st.caption(f"模型使用: {', '.join(feature_names['price_level'])}")
                 if price_level_pred_status == "error":
                     st.caption(f"错误: {price_level_error_msg}")


        with col3: # 均价预测
            st.markdown(f"<h5 style='color: {unit_price_color}; margin-bottom: 5px;'>均价预测</h5>", unsafe_allow_html=True)
            if regression_pred_status == "success":
                 # --- 修改点：移除标签，只显示值 ---
                 st.markdown(f"""
                    <div style='margin-bottom: 10px;'>
                        <p style='font-size: 28px; font-weight: bold; color: {unit_price_color}; margin-top: 0px;'>
                            {regression_pred_result:,.0f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            elif regression_pred_status == "insufficient_data":
                 st.warning("数据不足")
            elif regression_pred_status == "error":
                 st.error("预测出错")
            else: # not_run
                 st.markdown(f"<p style='font-size: small; color: {grey_color};'>未运行</p>", unsafe_allow_html=True)

            with st.expander("查看使用/缺失特征"):
                 st.info("提示：该预测通常不依赖'总价'输入。")
                 if regression_pred_status == "insufficient_data":
                    st.caption(f"需要但缺失: {', '.join(regression_missing_features)}")
                 elif feature_names.get('regression'):
                    st.caption(f"模型使用: {', '.join(feature_names['regression'])}")
                 if regression_pred_status == "error":
                     st.caption(f"错误: {regression_error_msg}")


        # --- 显示总体状态和运行时错误 ---
        st.markdown("---")
        if runtime_errors:
             st.error("预测过程中发生运行时错误：")
             for i, msg in enumerate(runtime_errors):
                 st.markdown(f"{i+1}. {msg}")
        elif any(status == "insufficient_data" for status in [market_pred_status, price_level_pred_status, regression_pred_status]):
            st.warning("部分预测因缺少必要输入数据而无法完成。请在侧边栏提供所需信息。")
        elif all(status == "success" or status == "not_run" for status in [market_pred_status, price_level_pred_status, regression_pred_status]):
             # 只有在没有运行时错误，并且没有“数据不足”的情况下才显示完全成功
             st.success("✅ 分析完成！请查看上方结果。")

        st.info("💡 **提示:** 预测结果仅供参考，实际交易价格受多重因素影响。")

# --- 页脚信息 ---
st.sidebar.markdown("---")
st.sidebar.caption("模型信息: LightGBM & RandomForest")
st.sidebar.caption("数据来源: 安居客")
st.sidebar.caption("开发者: 凌欢")