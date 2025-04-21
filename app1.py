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

# --- 常量定义：模型和资源文件路径 ---
MARKET_MODEL_PATH = 'market_segment_lgbm_model.joblib' # 市场细分模型
PRICE_LEVEL_MODEL_PATH = 'price_level_rf_model.joblib'   # 价格水平模型
REGRESSION_MODEL_PATH = 'unit_price_rf_model.joblib'    # 均价回归模型
SCALER_PATH = 'regression_scaler.joblib'             # 回归模型使用的Scaler
FEATURE_NAMES_PATH = 'feature_names.joblib'           # 各模型所需特征列表
MAPPINGS_PATH = 'mappings.joblib'                     # 分类特征编码映射

# --- 加载资源函数 (使用缓存) ---
@st.cache_resource # 使用 Streamlit 缓存，避免重复加载
def load_resources():
    """加载所有必要的资源文件 (模型, scaler, 特征名, 映射关系)。"""
    resources = {}
    all_files_exist = True
    required_files = [MARKET_MODEL_PATH, PRICE_LEVEL_MODEL_PATH, REGRESSION_MODEL_PATH,
                      SCALER_PATH, FEATURE_NAMES_PATH, MAPPINGS_PATH]
    missing_files = []
    # 检查所有必需文件是否存在
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 未找到。") # 在控制台打印错误
            missing_files.append(file_path)
            all_files_exist = False
    if not all_files_exist:
        print(f"错误：缺少文件 {missing_files}。请确保所有 .joblib 文件与 app.py 在同一目录。")
        return None, missing_files # 返回 None 表示加载失败，并附带缺失文件列表

    # 尝试加载文件
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
        return resources, None # 返回包含所有资源的字典，错误信息为 None
    except Exception as e:
        print(f"加载资源时发生错误: {e}") # 在控制台打印加载异常
        return None, [f"加载错误: {e}"] # 返回 None，并附带错误信息

# --- 执行资源加载 ---
resources, load_error_info = load_resources()

# --- 辅助函数 ---
def format_mapping_options_for_selectbox(name_to_code_mapping):
    """
    为 Streamlit Selectbox 准备选项和格式化函数所需的数据。
    输入: {'名称1': 代码1, '名称2': 代码2, ...}
    输出: {代码1: '名称1 (代码1)', 代码2: '名称2 (代码2)', ...} (按代码排序)
    """
    if not isinstance(name_to_code_mapping, dict):
        print(f"[格式化错误] 输入非字典: {type(name_to_code_mapping)}")
        return {} # 返回空字典以避免后续错误
    code_to_display_string = {}
    try:
        # 尝试将代码转换为整数进行排序，如果失败则按字符串排序
        try:
             sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        except ValueError:
             sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: str(item[1]))

        for name, code in sorted_items:
             try:
                 # 优先使用整数作为字典键
                 code_key = int(code)
                 name_str = str(name)
                 code_to_display_string[code_key] = f"{name_str} ({code_key})"
             except (ValueError, TypeError):
                 # 如果代码不能转为整数，使用字符串
                 code_key = str(code)
                 name_str = str(name)
                 code_to_display_string[code_key] = f"{name_str} ({code_key})"
        return code_to_display_string
    except Exception as e: # 捕获排序/转换过程中的其他潜在错误
        print(f"[格式化错误] 转换/排序时出错: {e}")
        # 如果排序失败，尝试不排序直接创建映射
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
欢迎使用盐城二手房分析工具！请在左侧边栏输入房产特征（可选项留空或不勾选表示未知），我们将为您提供三个维度的预测：
1.  **市场细分预测**: 判断房产属于低端、中端还是高端市场。
2.  **价格水平预测**: 判断房产单价是否高于其所在区域的平均水平。
3.  **房产均价预测**: 预测房产的每平方米单价（元/㎡）。
""")
st.markdown("---") # 分隔线

# --- 应用启动时资源加载失败或映射缺失的处理 ---
if not resources:
     st.error("❌ **应用程序初始化失败！**")
     if load_error_info:
         st.warning(f"无法加载必要的资源文件。错误详情:")
         for error in load_error_info:
             st.markdown(f"*   `{error}`")
     else:
         # 如果 load_resources 返回 (None, None) 或 (None, [])
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
        *   确保 `{MAPPINGS_PATH}` 和 `{FEATURE_NAMES_PATH}` 文件内容有效且包含所需键。
        *   检查运行 Streamlit 的终端是否有更详细的错误信息。
     """)
     st.stop() # 停止执行，因为无法继续

# --- 如果资源加载成功，进行内容校验 ---
mappings = resources['mappings']
feature_names = resources['feature_names']
market_model = resources['market_model']
price_level_model = resources['price_level_model']
regression_model = resources['regression_model']
scaler = resources['scaler']

# --- 检查资源文件内容是否符合预期 ---
required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价'] # 需要的映射名称
required_features_keys = ['market', 'price_level', 'regression'] # 需要的特征列表名称
valid_resources = True
missing_or_invalid = [] # 存储缺失或无效的资源项

# 检查映射文件
for key in required_mappings:
    if key not in mappings or not isinstance(mappings.get(key), dict):
        missing_or_invalid.append(f"映射 '{key}' (来自 {MAPPINGS_PATH})")
        valid_resources = False

# 检查特征名称文件
for key in required_features_keys:
    if key not in feature_names or not isinstance(feature_names.get(key), list):
        missing_or_invalid.append(f"特征列表 '{key}' (来自 {FEATURE_NAMES_PATH})")
        valid_resources = False

# 如果资源内容有问题，显示错误并停止
if not valid_resources:
    st.error(f"❌ **资源文件内容错误！**")
    st.warning("以下必需的映射或特征列表在资源文件中缺失、无效或格式错误：")
    for item in missing_or_invalid:
        st.markdown(f"*   {item}")
    st.warning(f"请检查 `{MAPPINGS_PATH}` 和 `{FEATURE_NAMES_PATH}` 文件内容。")
    st.stop() # 停止执行

# --- 侧边栏输入控件 ---
st.sidebar.header("🏘️ 房产特征输入")
st.sidebar.caption("对于不确定的特征，请保留默认选项或不勾选复选框。") # 提示用户如何表示未知

# --- 分类特征输入 (下拉选择框) ---
st.sidebar.subheader("选择项特征")
selectbox_inputs = {} # 存储下拉框选择结果
categorical_feature_keys = ['方位', '楼层', '所属区域', '房龄'] # 下拉框对应的特征名称

# 辅助函数，用于创建下拉选择框，包含“未知”选项
def create_selectbox(label, mapping_key, help_text, key_suffix):
    """创建包含 '--- 请选择 ---' (代表 None) 选项的下拉框。"""
    try:
        options_map = mappings[mapping_key] # 获取名称到代码的映射
        display_map = format_mapping_options_for_selectbox(options_map) # 获取代码到显示字符串的映射
        # 选项列表，将 None (代表未选择) 放在最前面
        options_codes = [None] + list(display_map.keys())

        # 定义选项的显示格式
        def format_func(x):
            if x is None:
                return "--- 请选择 ---" # 未选择时显示这个文本
            return display_map.get(x, f"未知代码 ({x})") # 已选择时显示 "名称 (代码)"

        # 创建 selectbox
        selectbox_inputs[mapping_key] = st.sidebar.selectbox(
            f"{label}:",
            options=options_codes,
            index=0, # 默认选中第一个选项，即 "--- 请选择 ---"
            format_func=format_func,
            key=f"{key_suffix}_select", # 每个控件需要唯一的 key
            help=help_text # 鼠标悬停时的帮助提示
        )
    except KeyError:
         st.sidebar.error(f"错误：映射文件中未找到 '{mapping_key}'。")
         selectbox_inputs[mapping_key] = None # 标记为错误状态
    except Exception as e:
        st.sidebar.error(f"{label} 选项加载错误: {e}")
        selectbox_inputs[mapping_key] = None # 标记为错误状态

# 创建各个下拉选择框
create_selectbox("房屋方位", '方位', "选择房屋的主要朝向。如果未知，请保留'请选择'。", "orientation")
create_selectbox("楼层位置", '楼层', "选择房屋所在的楼层范围（低、中、高）。如果未知，请保留'请选择'。", "floor_level")
create_selectbox("所属区域", '所属区域', "选择房产所在的盐城市主要区域。如果未知，请保留'请选择'。", "district")
create_selectbox("房龄范围", '房龄', "选择房屋的建造年限范围。如果未知，请保留'请选择'。", "age")


# --- 数值特征输入 (复选框 + 数字输入框) ---
st.sidebar.subheader("数值项特征")
numeric_inputs = {} # 存储最终有效的数值输入 (如果勾选了复选框)
numeric_widgets = {} # 临时存储 st.number_input 控件对象，以便后续读取值

# 辅助函数，创建 复选框 + 数字输入 的组合
def create_numeric_input(label, key, help_text, default_value, min_val, max_val, step, format_str):
    """创建带复选框的数字输入，未勾选时值为 None。"""
    # 使用列布局，让复选框和输入框在一行
    col1, col2 = st.sidebar.columns([1, 3]) # 调整列宽比例

    # 复选框，默认勾选 (value=True)
    provide_value = col1.checkbox("提供?", value=True, key=f"{key}_provide", help=f"勾选表示您将提供 '{label}' 的值。")

    # 数字输入框，只有在复选框勾选时才启用 (disabled=not provide_value)
    widget = col2.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=default_value,
        step=step,
        format=format_str,
        key=key, # 主键
        help=help_text,
        disabled=not provide_value # 根据复选框状态决定是否禁用
    )
    # 存储控件本身和复选框状态，以便按钮点击时读取
    numeric_widgets[key] = {'widget': widget, 'provide': provide_value}

# 创建各个数值输入控件
create_numeric_input("总价 (万)", '总价(万)', "输入房产的总价，单位万元。", 100.0, 10.0, 1500.0, 5.0, "%.1f")
create_numeric_input("面积 (㎡)", '面积(㎡)', "输入房产的建筑面积，单位平方米。", 100.0, 30.0, 600.0, 5.0, "%.1f")
create_numeric_input("建造时间 (年)", '建造时间', "输入房屋的建造年份。", 2018, 1970, 2025, 1, "%d")
create_numeric_input("总楼层数", '楼层数', "输入楼栋的总楼层数。", 18, 1, 60, 1, "%d")
create_numeric_input("室", '室', "输入卧室数量。", 3, 1, 10, 1, "%d")
create_numeric_input("厅", '厅', "输入客厅/餐厅数量。", 2, 0, 5, 1, "%d")
create_numeric_input("卫", '卫', "输入卫生间数量。", 1, 0, 5, 1, "%d")


# --- 预测触发按钮 ---
st.sidebar.markdown("---") # 侧边栏分隔线
if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help="点击这里根据输入的特征进行预测分析"):

    # --- 收集所有输入数据 ---
    # 读取数值输入的值，仅当复选框被勾选时
    numeric_inputs = {}
    for key, data in numeric_widgets.items():
        if data['provide']: # 如果勾选了 "提供?"
            # 从 Streamlit 的 session state 中读取 number_input 的当前值
            # Streamlit 更新 number_input 的值，即使它被禁用，所以我们需要从 state 读取
            numeric_inputs[key] = st.session_state[key]
        else:
            numeric_inputs[key] = None # 未勾选，值为 None

    # 合并下拉框和数值输入
    all_inputs = {**selectbox_inputs, **numeric_inputs}
    # print("收集到的所有输入:", all_inputs) # 调试用：打印所有输入值

    error_messages = [] # 存储预测过程中发生的错误信息
    prediction_results = {} # 存储每个模型的预测结果或状态信息

    # --- 获取各模型所需的特征名称 ---
    try:
        market_features_needed = feature_names['market']
        price_level_features_needed = feature_names['price_level']
        regression_features_needed = feature_names['regression']
    except KeyError as e:
        st.error(f"加载特征名称时出错: 必需的键 '{e}' 在 `feature_names.joblib` 文件中缺失。")
        st.stop() # 停止执行

    # --- 辅助函数：检查模型所需的输入是否都已提供 (不为 None) ---
    def check_missing_inputs(model_features, current_inputs):
        """检查指定模型所需的特征是否有缺失值 (None)。"""
        missing = []
        for feature in model_features:
            if current_inputs.get(feature) is None: # 使用 .get() 避免 KeyError
                missing.append(feature)
        # print(f"模型特征: {model_features}, 缺失: {missing}") # 调试用
        return missing

    # --- 1. 市场细分预测 ---
    market_pred_label = "待处理" # 初始状态
    # 检查市场细分模型所需的输入是否有缺失
    market_missing = check_missing_inputs(market_features_needed, all_inputs)
    if market_missing:
        # 如果有缺失，设置提示信息
        market_pred_label = f"数据不足，无法判断 (缺少: {', '.join(market_missing)})"
    else:
        # 如果输入完整，尝试进行预测
        try:
            # 准备输入数据，确保特征顺序与模型训练时一致
            input_data_market = {feat: all_inputs[feat] for feat in market_features_needed}
            input_df_market = pd.DataFrame([input_data_market])[market_features_needed] # 创建 DataFrame

            # 进行预测
            market_pred_code = market_model.predict(input_df_market)[0]

            # 获取编码到名称的反向映射 (如果 mappings 里存的是 name->code)
            # 假设 mappings['市场类别'] 是 {'低端': 0, '中端': 1, '高端': 2}
            market_code_to_name = {int(v): k for k, v in mappings.get('市场类别', {}).items()}
            market_pred_label = market_code_to_name.get(int(market_pred_code), f"预测编码无效 ({market_pred_code})")

        except Exception as e:
            msg = f"市场细分模型预测出错: {e}"
            print(msg) # 在控制台打印错误详情
            error_messages.append(msg) # 记录错误信息
            market_pred_label = "预测失败" # 设置失败状态
    prediction_results['market'] = market_pred_label # 存储结果

    # --- 2. 价格水平预测 ---
    price_level_pred_label = "待处理"
    price_level_pred_code = -1 # 默认代码，-1 表示未预测或失败
    # 检查价格水平模型所需的输入
    price_level_missing = check_missing_inputs(price_level_features_needed, all_inputs)
    if price_level_missing:
         price_level_pred_label = f"数据不足，无法判断 (缺少: {', '.join(price_level_missing)})"
    else:
        try:
            # 准备输入数据
            input_data_price_level = {feat: all_inputs[feat] for feat in price_level_features_needed}
            input_df_price_level = pd.DataFrame([input_data_price_level])[price_level_features_needed]

            # 进行预测
            price_level_pred_code = int(price_level_model.predict(input_df_price_level)[0])

            # 获取反向映射 (假设 mappings['是否高于区域均价'] 是 {'不高于': 0, '高于': 1})
            price_level_code_to_name = {int(v): k for k, v in mappings.get('是否高于区域均价', {}).items()}
            price_level_pred_label = price_level_code_to_name.get(price_level_pred_code, f"预测编码无效 ({price_level_pred_code})")

        except Exception as e:
            msg = f"价格水平模型预测出错: {e}"
            print(msg)
            error_messages.append(msg)
            price_level_pred_label = "预测失败"
            price_level_pred_code = -1 # 重置代码为失败状态
    prediction_results['price_level'] = price_level_pred_label
    prediction_results['price_level_code'] = price_level_pred_code # 存储预测代码，虽然不再用于颜色

    # --- 3. 回归预测 (均价) ---
    unit_price_pred_display = "待处理" # 用于显示的字符串
    unit_price_pred_value = -1.0    # 数值结果，-1.0 表示未计算或失败
    # 检查均价回归模型所需的输入
    regression_missing = check_missing_inputs(regression_features_needed, all_inputs)
    if regression_missing:
         unit_price_pred_display = f"数据不足，无法判断 (缺少: {', '.join(regression_missing)})"
    else:
        try:
            # 准备输入数据
            input_data_reg = {feat: all_inputs[feat] for feat in regression_features_needed}
            input_df_reg = pd.DataFrame([input_data_reg])[regression_features_needed]

            # 使用加载的 scaler 对数据进行标准化/归一化
            input_df_reg_scaled = scaler.transform(input_df_reg)

            # 进行预测
            unit_price_pred_raw = regression_model.predict(input_df_reg_scaled)[0]
            unit_price_pred_value = max(0.0, unit_price_pred_raw) # 确保预测价格不为负
            unit_price_pred_display = f"{unit_price_pred_value:,.0f}" # 格式化为整数，带千位分隔符

        except Exception as e:
            msg = f"均价预测模型预测出错: {e}"
            print(msg)
            error_messages.append(msg)
            unit_price_pred_display = "预测失败"
            unit_price_pred_value = -1.0 # 标记为失败
    prediction_results['unit_price_display'] = unit_price_pred_display
    prediction_results['unit_price_value'] = unit_price_pred_value # 存储数值结果

    # --- 结果显示区域 (左对齐，无描述，价格水平颜色固定) ---
    st.markdown("---") # 主页面分隔线
    st.subheader("📈 预测结果分析")

    # --- 定义结果显示的颜色 ---
    market_color = "#1f77b4"          # 市场细分标题和结果颜色 (蓝色)
    price_level_color_fixed = "#ff7f0e" # 价格水平标题和结果颜色 (橙色 - 固定)
    unit_price_color = "#2ca02c"       # 均价预测标题和结果颜色 (绿色)
    insufficient_data_color = "#7f7f7f" # 数据不足提示颜色 (灰色)
    error_color = "#d62728"             # 预测失败/错误提示颜色 (红色)

    # 使用列布局显示三个预测结果
    col1, col2, col3 = st.columns(3)

    # --- 第一列：市场细分 ---
    with col1:
        # 显示标题 (左对齐)
        st.markdown(f"<h5 style='color: {market_color}; margin-bottom: 5px;'>市场细分</h5>", unsafe_allow_html=True)
        market_result = prediction_results['market']
        # 根据预测结果状态设置颜色和样式
        if "数据不足" in market_result:
            display_color = insufficient_data_color
            font_size = "18px"
            font_weight = "normal"
        elif "失败" in market_result or "无效" in market_result:
            display_color = error_color
            font_size = "18px"
            font_weight = "bold"
        else: # 成功预测
            display_color = market_color
            font_size = "24px"
            font_weight = "bold"
        # 显示结果
        st.markdown(f"<p style='font-size: {font_size}; font-weight: {font_weight}; color: {display_color}; margin-bottom: 10px;'>{market_result}</p>", unsafe_allow_html=True)

    # --- 第二列：价格水平 ---
    with col2:
        # 显示标题 (左对齐，使用固定橙色)
        st.markdown(f"<h5 style='color: {price_level_color_fixed}; margin-bottom: 5px;'>价格水平 (相对区域)</h5>", unsafe_allow_html=True)
        price_level_result = prediction_results['price_level']
        # 根据预测结果状态设置颜色和样式 (注意：成功时的颜色固定为 price_level_color_fixed)
        if "数据不足" in price_level_result:
             display_color = insufficient_data_color
             font_size = "18px"
             font_weight = "normal"
        elif "失败" in price_level_result or "无效" in price_level_result:
             display_color = error_color
             font_size = "18px"
             font_weight = "bold"
        else: # 成功预测
             display_color = price_level_color_fixed # 结果颜色与标题一致 (固定橙色)
             font_size = "24px"
             font_weight = "bold"
        # 显示结果
        st.markdown(f"<p style='font-size: {font_size}; font-weight: {font_weight}; color: {display_color}; margin-bottom: 10px;'>{price_level_result}</p>", unsafe_allow_html=True)

    # --- 第三列：均价预测 ---
    with col3:
        # 显示标题 (左对齐)
        st.markdown(f"<h5 style='color: {unit_price_color}; margin-bottom: 5px;'>均价预测</h5>", unsafe_allow_html=True)
        unit_price_result = prediction_results['unit_price_display']
        unit_price_value = prediction_results['unit_price_value']
        # 根据预测结果状态设置颜色和样式
        if "数据不足" in unit_price_result:
            display_color = insufficient_data_color
            font_size = "18px"
            font_weight = "normal"
            display_text = unit_price_result # 直接显示 "数据不足..."
        elif "失败" in unit_price_result or unit_price_value < 0: # 检查是否失败
            display_color = error_color
            font_size = "18px"
            font_weight = "bold"
            display_text = unit_price_result # 显示 "预测失败"
        else: # 成功预测
            display_color = unit_price_color
            font_size = "24px"
            font_weight = "bold"
            # 成功时，在数字后添加单位
            display_text = f"{unit_price_result} <span style='font-size: small; color: grey;'>元/㎡</span>"
        # 显示结果 (使用 display_text，可能包含 HTML)
        st.markdown(f"<p style='font-size: {font_size}; font-weight: {font_weight}; color: {display_color}; margin-bottom: 10px;'>{display_text}</p>", unsafe_allow_html=True)


    # --- 显示最终状态信息 ---
    st.markdown("---") # 分隔线
    if not error_messages: # 如果预测过程中没有抛出异常
         # 检查是否有因为数据不足而未能预测的情况
         insufficient_data_count = sum(1 for res in prediction_results.values() if isinstance(res, str) and "数据不足" in res)
         if insufficient_data_count > 0:
              # 如果有部分预测因数据不足未完成
              st.info(f"✅ 分析已尝试。部分预测因缺少必要的输入特征而无法完成（显示为“数据不足”），请补充输入后重试。")
         else:
              # 如果所有预测都成功完成
              st.success("✅ 所有分析预测已完成！请查看上方结果。")
         # 统一的提示信息
         st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
    else:
         # 如果预测过程中出现了错误 (Exception)
         st.warning("⚠️ 分析过程中遇到错误，部分或全部预测未能完成。")
         # 显示具体的错误信息
         for msg in error_messages:
              st.error(f"错误详情: {msg}")

# --- 页脚信息 ---
st.sidebar.markdown("---") # 侧边栏分隔线
st.sidebar.caption("模型信息: LightGBM & RandomForest")
st.sidebar.caption("数据来源: 安居客")
st.sidebar.caption("开发者: 凌欢")