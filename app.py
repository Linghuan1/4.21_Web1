# -*- coding: utf-8 -*- # 指定编码为 UTF-8
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 页面基础配置 ---
st.set_page_config(
    page_title="盐城二手房智能分析器",  # 设置浏览器标签页标题
    page_icon="🏠",                 # 设置浏览器标签页图标
    layout="wide",                 # 设置页面布局为宽屏模式
    initial_sidebar_state="expanded" # 设置侧边栏默认展开
)

# --- 常量定义：模型和资源文件路径 (假设在当前目录下) ---
# (保持不变)
MARKET_MODEL_PATH = 'market_segment_lgbm_model.joblib'
PRICE_LEVEL_MODEL_PATH = 'price_level_rf_model.joblib'
REGRESSION_MODEL_PATH = 'unit_price_rf_model.joblib'
SCALER_PATH = 'regression_scaler.joblib'
FEATURE_NAMES_PATH = 'feature_names.joblib'
MAPPINGS_PATH = 'mappings.joblib'

# --- 加载资源函数 (使用缓存) ---
# (保持不变)
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
# (保持不变)
def format_mapping_options_for_selectbox(name_to_code_mapping):
    """为 Streamlit Selectbox 准备选项和格式化函数所需的数据。"""
    if not isinstance(name_to_code_mapping, dict):
        print(f"[格式化错误] 输入非字典: {type(name_to_code_mapping)}")
        return {}
    code_to_display_string = {}
    try:
        sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        for name, code in sorted_items:
            code_int = int(code)
            name_str = str(name)
            code_to_display_string[code_int] = f"{name_str} ({code_int})"
        return code_to_display_string
    except (ValueError, TypeError, KeyError) as e:
        print(f"[格式化错误] 转换/排序时出错: {e}")
        return {int(v): f"{k} ({int(v)})" for k, v in name_to_code_mapping.items() if isinstance(v, (int, float, str)) and str(v).isdigit()}


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
# (保持不变)
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
            *   `{MARKET_MODEL_PATH}` ... (其他文件) ... `{MAPPINGS_PATH}`
        *   确保 `{MAPPINGS_PATH}` 和 `{FEATURE_NAMES_PATH}` 文件内容有效。
        *   检查运行 Streamlit 的终端是否有更详细的错误信息。
     """)
     st.stop()

# --- 如果资源加载成功 ---
# (从这里开始的逻辑保持不变，直到结果显示部分)
mappings = resources['mappings']
feature_names = resources['feature_names']
market_model = resources['market_model']
price_level_model = resources['price_level_model']
regression_model = resources['regression_model']
scaler = resources['scaler']

required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价']
required_features = ['market', 'price_level', 'regression']
valid_resources = True
for key in required_mappings:
    if key not in mappings or not isinstance(mappings[key], dict):
        st.error(f"错误：映射文件 `{MAPPINGS_PATH}` 缺少或无效: '{key}'")
        valid_resources = False
for key in required_features:
    if key not in feature_names or not isinstance(feature_names[key], list):
        st.error(f"错误：特征名称文件 `{FEATURE_NAMES_PATH}` 缺少或无效: '{key}'")
        valid_resources = False
if not valid_resources:
    st.warning("资源文件内容不完整或格式错误。")
    st.stop()

# --- 侧边栏输入控件 ---
# (保持不变)
st.sidebar.header("🏘️ 房产特征输入")
st.sidebar.subheader("选择项特征")
selectbox_inputs = {}
try:
    orientation_map = mappings['方位']
    orientation_display_map = format_mapping_options_for_selectbox(orientation_map)
    orientation_codes = list(orientation_display_map.keys())
    selectbox_inputs['方位'] = st.sidebar.selectbox("房屋方位:", options=orientation_codes, format_func=lambda x: orientation_display_map.get(x, f"未知 ({x})"), key="orientation_select", help="选择房屋的主要朝向。")
except Exception as e: st.sidebar.error(f"方位选项错误: {e}"); selectbox_inputs['方位'] = None
# ... (其他下拉框保持不变) ...
try:
    age_map = mappings['房龄']
    age_display_map = format_mapping_options_for_selectbox(age_map)
    age_codes = list(age_display_map.keys())
    selectbox_inputs['房龄'] = st.sidebar.selectbox("房龄:", options=age_codes, format_func=lambda x: age_display_map.get(x, f"未知 ({x})"), key="age_select", help="选择房屋的建造年限范围。")
except Exception as e: st.sidebar.error(f"房龄选项错误: {e}"); selectbox_inputs['房龄'] = None

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
    if None in selectbox_inputs.values():
        st.error("⚠️ **输入错误：** 请确保所有下拉选择框都有有效的选项。")
    else:
        # --- 准备输入数据和预测 ---
        # (这部分逻辑保持不变)
        prediction_possible = True
        error_messages = []
        market_pred_label = "未进行预测"
        price_level_pred_label = "未进行预测"
        price_level_pred_code = -1
        unit_price_pred = -1
        all_inputs = {**selectbox_inputs, **numeric_inputs}

        # 1. 市场细分预测
        try:
            market_features_needed = feature_names['market']
            input_data_market = {feat: all_inputs[feat] for feat in market_features_needed if feat in all_inputs}
            if len(input_data_market) != len(market_features_needed): raise KeyError(f"市场细分模型缺少输入特征: {set(market_features_needed) - set(input_data_market.keys())}")
            input_df_market = pd.DataFrame([input_data_market])[market_features_needed]
            market_pred_code = market_model.predict(input_df_market)[0]
            market_output_map = mappings.get('市场类别', {})
            market_pred_label = market_output_map.get(int(market_pred_code), f"预测编码无效 ({market_pred_code})")
        except Exception as e: msg = f"市场细分模型预测出错: {e}"; print(msg); error_messages.append(msg); prediction_possible = False

        # 2. 价格水平预测
        if prediction_possible:
            try:
                price_level_features_needed = feature_names['price_level']
                input_data_price_level = {feat: all_inputs[feat] for feat in price_level_features_needed if feat in all_inputs}
                if len(input_data_price_level) != len(price_level_features_needed): raise KeyError(f"价格水平模型缺少输入特征: {set(price_level_features_needed) - set(input_data_price_level.keys())}")
                input_df_price_level = pd.DataFrame([input_data_price_level])[price_level_features_needed]
                price_level_pred_code = price_level_model.predict(input_df_price_level)[0]
                price_level_output_map = mappings.get('是否高于区域均价', {})
                price_level_pred_label = price_level_output_map.get(int(price_level_pred_code), f"预测编码无效 ({price_level_pred_code})")
            except Exception as e: msg = f"价格水平模型预测出错: {e}"; print(msg); error_messages.append(msg); prediction_possible = False

        # 3. 回归预测
        if prediction_possible:
            try:
                regression_features_needed = feature_names['regression']
                input_data_reg = {feat: all_inputs[feat] for feat in regression_features_needed if feat in all_inputs}
                if len(input_data_reg) != len(regression_features_needed): raise KeyError(f"均价预测模型缺少输入特征: {set(regression_features_needed) - set(input_data_reg.keys())}")
                input_df_reg = pd.DataFrame([input_data_reg])[regression_features_needed]
                input_df_reg_scaled = scaler.transform(input_df_reg)
                unit_price_pred = regression_model.predict(input_df_reg_scaled)[0]
                unit_price_pred = max(0, unit_price_pred)
            except Exception as e: msg = f"均价预测模型预测出错: {e}"; print(msg); error_messages.append(msg); unit_price_pred = -1; prediction_possible = False # 标记为-1表示出错

        # --- 结果显示区域 (应用美化) ---
        if not error_messages:
            st.markdown("---")
            st.subheader("📈 预测结果分析")

            # 定义颜色以便复用
            market_color = "#1f77b4"  # 蓝色
            price_level_color = "#ff7f0e" # 橙色
            unit_price_color = "#2ca02c" # 绿色

            col1, col2, col3 = st.columns(3)

            with col1: # 市场细分 - 居中显示
                # 标题居中
                st.markdown(f"<h5 style='text-align: center; color: {market_color}; margin-bottom: 5px;'>市场细分</h5>", unsafe_allow_html=True)
                # st.markdown("<hr style='margin-top: 0px; margin-bottom: 10px;'>", unsafe_allow_html=True) # 分隔线可选

                # 结果居中
                if "错误" not in market_pred_label and "无效" not in market_pred_label and market_pred_label != "未进行预测":
                    st.markdown(f"<p style='text-align: center; font-size: 28px; font-weight: bold; color: {market_color}; margin-bottom: 10px;'>{market_pred_label}</p>", unsafe_allow_html=True)
                else:
                    st.warning(f"市场细分预测失败: {market_pred_label}")

                # 描述文字居中，使用 markdown 控制样式
                st.markdown("<p style='text-align: center; font-size: small; color: grey;'>判断房产在整体市场中的<br>价格定位。</p>", unsafe_allow_html=True) # 使用 <br> 进行换行

            with col2: # 价格水平 - 居中显示
                # 标题居中
                st.markdown(f"<h5 style='text-align: center; color: {price_level_color}; margin-bottom: 5px;'>价格水平 (相对区域)</h5>", unsafe_allow_html=True)
                # st.markdown("<hr style='margin-top: 0px; margin-bottom: 10px;'>", unsafe_allow_html=True)

                # 结果居中
                if "错误" not in price_level_pred_label and "无效" not in price_level_pred_label and price_level_pred_label != "未进行预测":
                     if price_level_pred_code == 1: display_text, display_color = price_level_pred_label, "#E74C3C" # 高于 (红色)
                     elif price_level_pred_code == 0: display_text, display_color = price_level_pred_label, "#2ECC71" # 不高于 (绿色)
                     else: display_text, display_color = "未知状态", "#7f7f7f" # 灰色表示未知
                     st.markdown(f"<p style='text-align: center; font-size: 28px; font-weight: bold; color: {display_color}; margin-bottom: 10px;'>{display_text}</p>", unsafe_allow_html=True)
                else:
                     st.warning(f"价格水平预测失败: {price_level_pred_label}")

                # 描述文字居中
                st.markdown("<p style='text-align: center; font-size: small; color: grey;'>判断房产单价是否高于<br>其所在区域均值。</p>", unsafe_allow_html=True)

            with col3: # 均价预测 - 标题、描述居中，描述颜色匹配标题
                # 标题居中
                st.markdown(f"<h5 style='text-align: center; color: {unit_price_color}; margin-bottom: 5px;'>均价预测</h5>", unsafe_allow_html=True)
                # st.markdown("<hr style='margin-top: 0px; margin-bottom: 10px;'>", unsafe_allow_html=True)

                # 结果 (st.metric 默认左对齐，但视觉上通常可接受)
                if unit_price_pred != -1: # 检查是否成功预测
                     # st.metric 本身不易完全居中，但其内部标签和值对齐
                     # 为了视觉上更协调，可以在 metric 外面包一个 div 并尝试居中，但这会更复杂
                     # 简单的做法是接受 st.metric 的默认对齐
                     st.metric(label="预测单价 (元/㎡)", value=f"{unit_price_pred:,.0f}")
                else:
                     st.warning("无法完成房产均价预测。")

                # 描述文字居中，颜色与标题一致
                if unit_price_pred != -1: # 只有成功预测才显示描述
                    st.markdown(f"<p style='text-align: center; font-size: small; color: {unit_price_color};'>预测的每平方米<br>大致价格。</p>", unsafe_allow_html=True)

            # 整体成功提示
            st.success("✅ 分析预测完成！请查看上方结果。")
            st.markdown("---")
            st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
        else:
            # 如果有错误信息
            st.warning("部分或全部预测因输入或模型准备错误未能完成，请检查具体错误信息。")
            for msg in error_messages:
                st.error(msg) # 显示具体的错误信息

# --- 页脚信息 ---
# (保持不变)
st.sidebar.markdown("---")
st.sidebar.caption("模型信息: LightGBM & RandomForest")
st.sidebar.caption("数据来源: 安居客 (示例)")
st.sidebar.caption("开发者: 凌欢")