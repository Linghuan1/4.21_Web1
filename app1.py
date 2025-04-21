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
MARKET_MODEL_PATH = 'market_segment_lgbm_model.joblib'
PRICE_LEVEL_MODEL_PATH = 'price_level_rf_model.joblib'
REGRESSION_MODEL_PATH = 'unit_price_rf_model.joblib'
SCALER_PATH = 'regression_scaler.joblib'
FEATURE_NAMES_PATH = 'feature_names.joblib'
MAPPINGS_PATH = 'mappings.joblib'

# --- 加载资源函数 (使用缓存) ---
@st.cache_resource # 使用 Streamlit 的缓存机制，避免重复加载
def load_resources():
    """加载所有必要的资源文件 (模型, scaler, 特征名, 映射关系)。"""
    resources = {}
    all_files_exist = True
    # 需要加载的文件列表
    required_files = [MARKET_MODEL_PATH, PRICE_LEVEL_MODEL_PATH, REGRESSION_MODEL_PATH,
                      SCALER_PATH, FEATURE_NAMES_PATH, MAPPINGS_PATH]
    missing_files = []
    # 检查所有文件是否存在
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 未找到。")
            missing_files.append(file_path)
            all_files_exist = False
    if not all_files_exist:
        print(f"错误：缺少文件 {missing_files}。请确保所有 .joblib 文件与 app.py 在同一目录。")
        return None, missing_files # 返回 None 表示加载失败，并附带缺失文件列表

    # 尝试加载文件
    try:
        resources['market_model'] = joblib.load(MARKET_MODEL_PATH)         # 市场细分模型
        resources['price_level_model'] = joblib.load(PRICE_LEVEL_MODEL_PATH) # 价格水平模型
        resources['regression_model'] = joblib.load(REGRESSION_MODEL_PATH)   # 回归预测模型
        resources['scaler'] = joblib.load(SCALER_PATH)                     # 回归模型用的数据缩放器
        resources['feature_names'] = joblib.load(FEATURE_NAMES_PATH)       # 各模型所需的特征名列表
        resources['mappings'] = joblib.load(MAPPINGS_PATH)                 # 分类特征的编码映射关系
        print("所有资源加载成功。")
        print("从文件加载的映射关系:", resources['mappings'])
        print("从文件加载的特征名称:", resources['feature_names'])
        return resources, None # 返回加载的资源和 None 表示无错误
    except Exception as e:
        print(f"加载资源时发生错误: {e}")
        return None, [f"加载错误: {e}"] # 返回 None 和错误信息

# --- 辅助函数：格式化下拉框选项 ---
def format_mapping_options_for_selectbox(name_to_code_mapping):
    """为 Streamlit Selectbox 准备选项和格式化函数所需的数据。"""
    if not isinstance(name_to_code_mapping, dict):
        print(f"[格式化错误] 输入非字典: {type(name_to_code_mapping)}")
        return {} # 返回空字典表示失败
    code_to_display_string = {}
    try:
        # 尝试将 code 转换为 int 进行排序，如果失败则按字符串排序
        try:
            # 按编码值（通常是数字）排序，方便用户查找
            sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        except ValueError:
             print(f"[格式化警告] 无法将所有 code 转换为 int 进行排序，将按字符串排序: {name_to_code_mapping}")
             sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: str(item[1]))

        # 创建用于显示的 编码 -> "名称 (编码)" 映射
        for name, code in sorted_items:
            try:
                code_key = int(code) # Selectbox 的选项通常需要原始类型作为键
            except ValueError:
                code_key = str(code) # 如果不能转为int，则保留字符串类型
            name_str = str(name)
            code_to_display_string[code_key] = f"{name_str} ({code})" # 显示格式：名称 (编码)
        return code_to_display_string
    except (TypeError, KeyError) as e:
        print(f"[格式化错误] 格式化选项时出错: {e}")
        # 备选方案：尝试直接用原始键值对，处理可能出现的异常
        fallback_map = {}
        for k, v in name_to_code_mapping.items():
             try:
                 fallback_map[v] = f"{k} ({v})"
             except Exception: # 捕获潜在的哈希错误等
                 pass
        return fallback_map


# --- 加载资源 ---
resources, load_error_info = load_resources()

# --- Streamlit 用户界面主要部分 ---
st.title("🏠 盐城二手房智能分析与预测")
st.markdown("""
欢迎使用盐城二手房分析工具！请在左侧边栏输入房产特征，我们将为您提供三个维度的预测：
1.  **市场细分预测**: 判断房产属于低端、中端还是高端市场。
2.  **价格水平预测**: 判断房产单价是否高于其所在区域的平均水平。
3.  **房产均价预测**: 预测房产的每平方米单价（元/㎡）。

👇 下方的预测结果中，您可以展开 **"使用特征"** 查看每个预测模型依赖的具体输入项。
""")
st.markdown("---") # 分隔线

# --- 应用启动时资源加载失败或映射缺失的处理 ---
if not resources:
     st.error("❌ **应用程序初始化失败！**")
     if load_error_info:
         st.warning(f"无法加载必要的资源文件。错误详情:")
         for error in load_error_info:
             st.markdown(f"*   `{error}`") # 显示加载错误信息
     else:
         st.warning("无法找到一个或多个必需的资源文件。")
     # 提供用户检查指引
     st.markdown(f"""
        请检查以下几点：
        *   确认以下所有 `.joblib` 文件都与 `app.py` 文件在 **同一个** 目录下:
            *   `{MARKET_MODEL_PATH}`
            *   `{PRICE_LEVEL_MODEL_PATH}`
            *   `{REGRESSION_MODEL_PATH}`
            *   `{SCALER_PATH}`
            *   `{FEATURE_NAMES_PATH}`
            *   `{MAPPINGS_PATH}`
        *   确保 `{MAPPINGS_PATH}` 和 `{FEATURE_NAMES_PATH}` 文件内容有效且格式正确。
        *   检查运行 Streamlit 的终端/控制台是否有更详细的错误信息。
     """)
     st.stop() # 停止应用执行

# --- 如果资源加载成功 ---
# 从加载的资源中提取对象
mappings = resources['mappings']
feature_names = resources['feature_names']
market_model = resources['market_model']
price_level_model = resources['price_level_model']
regression_model = resources['regression_model']
scaler = resources['scaler']

# 检查核心映射和特征列表是否存在且为预期类型，增强鲁棒性
required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价'] # 需要用到的映射
required_features = ['market', 'price_level', 'regression'] # 需要用到的特征列表键名
valid_resources = True
missing_or_invalid = [] # 记录缺失或无效的项目

# 检查映射文件内容
for key in required_mappings:
    if key not in mappings or not isinstance(mappings.get(key), dict):
        missing_or_invalid.append(f"映射 '{key}' (来自 {MAPPINGS_PATH})")
        valid_resources = False

# 检查特征名称文件内容
for key in required_features:
    if key not in feature_names or not isinstance(feature_names.get(key), list):
        missing_or_invalid.append(f"特征列表 '{key}' (来自 {FEATURE_NAMES_PATH})")
        valid_resources = False

# 如果资源检查失败，则提示并停止
if not valid_resources:
    st.error(f"❌ 资源文件内容不完整或格式错误。缺少或无效的项目:")
    for item in missing_or_invalid:
        st.markdown(f"*   {item}")
    st.stop()

# --- 侧边栏输入控件 ---
st.sidebar.header("🏘️ 房产特征输入")
st.sidebar.subheader("选择项特征")
selectbox_inputs = {} # 存储下拉框输入的值
all_select_valid = True # 标志：追踪下拉框是否都成功加载

# --- 封装下拉框创建逻辑 ---
def create_selectbox(label, mapping_key, help_text, key_suffix):
    """创建并返回一个 Streamlit 下拉选择框的值，处理潜在错误。"""
    global all_select_valid # 允许修改全局标志
    try:
        options_map = mappings[mapping_key] # 获取原始映射
        display_map = format_mapping_options_for_selectbox(options_map) # 格式化选项用于显示
        if not display_map: # 如果格式化后为空字典，说明有问题
             raise ValueError(f"映射 '{mapping_key}' 格式化后为空字典。原始: {options_map}")
        options_codes = list(display_map.keys()) # 获取所有选项的编码（作为 selectbox 的内部值）

        # 设置默认选中项：尝试选择中间或常见的选项，增加用户体验
        default_index = 0
        if options_codes:
            # 预设一些常见特征的默认值代码（需要根据实际映射调整）
            common_defaults = {'楼层': 1, '房龄': 2} # 假设 1=中楼层, 2=次新(5-10年)
            if mapping_key in common_defaults and common_defaults[mapping_key] in options_codes:
                 default_value = common_defaults[mapping_key]
                 try:
                    default_index = options_codes.index(default_value)
                 except ValueError: # 如果预设值不在选项中，则忽略
                    pass
            elif len(options_codes) > 1:
                 default_index = len(options_codes) // 2 # 否则选中间的选项

        # 创建 selectbox
        selected_value = st.sidebar.selectbox(
            label,
            options=options_codes, # 选项列表（编码）
            index=default_index,   # 默认选中的索引
            format_func=lambda x: display_map.get(x, f"未知选项 ({x})"), # 显示格式：名称 (编码)
            key=f"{key_suffix}_select", # 唯一键
            help=help_text # 帮助提示
        )
        return selected_value
    except Exception as e:
        st.sidebar.error(f"加载 '{label}' 选项时出错: {e}") # 显示错误信息
        all_select_valid = False # 标记为失败
        return None # 返回 None 表示失败

# --- 创建各个下拉选择框 ---
selectbox_inputs['方位'] = create_selectbox("房屋方位:", '方位', "选择房屋的主要朝向。", "orientation")
selectbox_inputs['楼层'] = create_selectbox("楼层位置:", '楼层', "选择房屋所在楼层的大致位置（低、中、高）。", "floor_level")
selectbox_inputs['所属区域'] = create_selectbox("所属区域:", '所属区域', "选择房产所在的行政区域或板块。", "district")
selectbox_inputs['房龄'] = create_selectbox("房龄:", '房龄', "选择房屋的建造年限范围。", "age")

# --- 数值输入控件 ---
st.sidebar.subheader("数值项特征")
numeric_inputs = {} # 存储数值输入的值
# 创建 number_input 控件
numeric_inputs['总价(万)'] = st.sidebar.number_input("总价 (万):", min_value=10.0, max_value=2000.0, value=120.0, step=5.0, format="%.1f", key="total_price", help="输入房产的总价，单位万元。")
numeric_inputs['面积(㎡)'] = st.sidebar.number_input("面积 (㎡):", min_value=20.0, max_value=800.0, value=95.0, step=1.0, format="%.1f", key="area_sqm", help="输入房产的建筑面积，单位平方米。")
numeric_inputs['建造时间'] = st.sidebar.number_input("建造时间 (年份):", min_value=1970, max_value=2024, value=2015, step=1, format="%d", key="build_year", help="输入房屋的建造年份。")
numeric_inputs['楼层数'] = st.sidebar.number_input("总楼层数:", min_value=1, max_value=70, value=18, step=1, format="%d", key="floor_num", help="输入楼栋的总楼层数。")
numeric_inputs['室'] = st.sidebar.number_input("室:", min_value=1, max_value=10, value=3, step=1, format="%d", key="rooms", help="输入卧室数量。")
numeric_inputs['厅'] = st.sidebar.number_input("厅:", min_value=0, max_value=5, value=2, step=1, format="%d", key="halls", help="输入客厅/餐厅数量。")
numeric_inputs['卫'] = st.sidebar.number_input("卫:", min_value=0, max_value=6, value=1, step=1, format="%d", key="baths", help="输入卫生间数量。")

# --- 预测触发按钮 ---
st.sidebar.markdown("---") # 侧边栏分隔线
# 只有在所有下拉框都成功加载时才启用按钮，否则禁用
predict_button_disabled = not all_select_valid
predict_button_help = "点击这里根据输入的特征进行预测分析" if all_select_valid else "部分下拉框选项加载失败，无法进行预测。请检查资源文件或错误信息。"

# 创建按钮，状态由 all_select_valid 控制
if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help=predict_button_help, disabled=predict_button_disabled):
    # 再次检查下拉框是否有 None 值 (以防万一)
    if None in selectbox_inputs.values():
        st.error("⚠️ **输入错误：** 检测到无效的下拉选择项。请检查侧边栏是否有错误提示。")
    else:
        # --- 准备输入数据和预测 ---
        prediction_possible = True # 标志：预测是否可以继续
        error_messages = []        # 存储预测过程中发生的错误信息
        market_pred_label = "处理中..." # 市场细分预测结果标签初始化
        price_level_pred_label = "处理中..."# 价格水平预测结果标签初始化
        price_level_pred_code = -99       # 价格水平预测结果编码初始化（-99表示未预测或错误）
        unit_price_pred = -1.0            # 均价预测结果初始化 (-1.0表示未预测或错误)

        # 存储各模型实际使用的特征列表，用于后续展示
        market_features_used = []
        price_level_features_used = []
        regression_features_used = []

        # 合并所有输入项到一个字典
        all_inputs = {**selectbox_inputs, **numeric_inputs}
        print("准备预测的输入数据:", all_inputs) # 调试信息：打印所有输入值

        # --- 1. 市场细分预测 ---
        try:
            market_features_needed = feature_names['market'] # 获取模型需要的特征列表
            market_features_used = market_features_needed    # 记录使用的特征
            print("市场细分模型所需特征:", market_features_needed)
            input_data_market = {}
            missing_market_feats = [] # 记录缺失的特征
            # 检查并收集输入数据
            for feat in market_features_needed:
                if feat in all_inputs:
                    input_data_market[feat] = all_inputs[feat]
                else:
                    missing_market_feats.append(feat) # 如果特征在输入中找不到，则记录下来
            # 如果有缺失特征，则抛出错误
            if missing_market_feats:
                 raise KeyError(f"市场细分模型缺少输入特征: {', '.join(missing_market_feats)}")

            # 创建 DataFrame 并保证特征顺序与模型训练时一致
            input_df_market = pd.DataFrame([input_data_market])[market_features_needed]
            print("市场细分模型输入 DataFrame:", input_df_market)
            # 进行预测
            market_pred_code = market_model.predict(input_df_market)[0]
            print(f"市场细分预测原始编码: {market_pred_code}, 类型: {type(market_pred_code)}")
            # 获取输出标签映射
            market_output_map_raw = mappings.get('市场类别', {})
            # 将预测编码转换为正确的键类型（int 或 str）来查找标签
            market_pred_key = int(market_pred_code) if isinstance(market_pred_code, (int, np.integer)) else str(market_pred_code)
            market_pred_label = market_output_map_raw.get(market_pred_key, f"未知编码 ({market_pred_key})") # 获取标签，如果找不到则显示未知
            print(f"市场细分预测标签: {market_pred_label}")
        except Exception as e:
            msg = f"市场细分模型预测出错: {e}"
            print(msg) # 打印错误信息到控制台
            error_messages.append(msg) # 收集错误信息用于界面显示
            market_pred_label = "预测失败" # 标记预测失败
            prediction_possible = False # 通常市场细分失败，后续预测意义不大或无法进行

        # --- 2. 价格水平预测 (仅在之前成功时进行) ---
        if prediction_possible: # 如果前面的预测成功
            try:
                price_level_features_needed = feature_names['price_level']
                price_level_features_used = price_level_features_needed # 记录使用的特征
                print("价格水平模型所需特征:", price_level_features_needed)
                input_data_price_level = {}
                missing_price_feats = []
                # 检查并收集输入数据
                for feat in price_level_features_needed:
                    if feat in all_inputs:
                         input_data_price_level[feat] = all_inputs[feat]
                    else:
                         missing_price_feats.append(feat)
                # 如果有缺失特征，则抛出错误
                if missing_price_feats:
                    raise KeyError(f"价格水平模型缺少输入特征: {', '.join(missing_price_feats)}")

                # 创建 DataFrame 并保证特征顺序
                input_df_price_level = pd.DataFrame([input_data_price_level])[price_level_features_needed]
                print("价格水平模型输入 DataFrame:", input_df_price_level)
                # 进行预测
                price_level_pred_code = price_level_model.predict(input_df_price_level)[0]
                print(f"价格水平预测原始编码: {price_level_pred_code}, 类型: {type(price_level_pred_code)}")
                # 获取输出标签映射
                price_level_output_map_raw = mappings.get('是否高于区域均价', {})
                # 转换编码为正确类型
                price_level_pred_key = int(price_level_pred_code) if isinstance(price_level_pred_code, (int, np.integer)) else str(price_level_pred_code)
                price_level_pred_label = price_level_output_map_raw.get(price_level_pred_key, f"未知编码 ({price_level_pred_key})")
                # 保留整数编码用于后续判断颜色，如果不是数字则标记为错误码
                if isinstance(price_level_pred_code, (int, np.integer)):
                    price_level_pred_code = int(price_level_pred_code)
                else:
                    price_level_pred_code = -99 # 无效编码
                print(f"价格水平预测标签: {price_level_pred_label}, 使用编码: {price_level_pred_code}")

            except Exception as e:
                msg = f"价格水平模型预测出错: {e}"
                print(msg)
                error_messages.append(msg)
                price_level_pred_label = "预测失败"
                price_level_pred_code = -99 # 标记为错误
                # 注意：这里不设置 prediction_possible = False，允许继续尝试回归预测

        # --- 3. 回归预测 (均价预测) ---
        # 尝试进行回归预测，即使价格水平预测失败（假设回归模型不依赖价格水平预测结果）
        regression_attempted = False # 标志：是否尝试了回归预测
        try:
            regression_features_needed = feature_names['regression']
            regression_features_used = regression_features_needed # 记录使用的特征
            print("均价预测模型所需特征:", regression_features_needed)
            # --- 注意：检查这里需要的特征，例如，如果不需要 '总价(万)'，它不应该出现在 regression_features_needed 列表中 ---
            # 检查并收集输入数据
            input_data_reg = {}
            missing_reg_feats = []
            for feat in regression_features_needed:
                if feat in all_inputs:
                    input_data_reg[feat] = all_inputs[feat]
                else:
                    missing_reg_feats.append(feat)
            # 如果有缺失特征，则抛出错误
            if missing_reg_feats:
                raise KeyError(f"均价预测模型缺少输入特征: {', '.join(missing_reg_feats)}")

            # 创建 DataFrame 并保证特征顺序
            input_df_reg = pd.DataFrame([input_data_reg])[regression_features_needed]
            print("均价预测模型输入 DataFrame (原始):", input_df_reg)
            # 应用数据缩放器 (scaler)
            input_df_reg_scaled = scaler.transform(input_df_reg)
            print("均价预测模型输入 DataFrame (缩放后):", input_df_reg_scaled)
            # 进行预测
            unit_price_pred = regression_model.predict(input_df_reg_scaled)[0]
            # 确保预测结果非负，并转换为 float 类型
            unit_price_pred = max(0, float(unit_price_pred))
            print(f"均价预测结果: {unit_price_pred}")
            regression_attempted = True # 标记已成功尝试

        except Exception as e:
            msg = f"均价预测模型预测出错: {e}"
            print(msg)
            error_messages.append(msg)
            unit_price_pred = -1.0 # 标记为错误

        # --- 结果显示区域 ---
        st.markdown("---") # 主页面分隔线
        st.subheader("📈 预测结果分析")

        # 定义结果区域的颜色
        market_color = "#1f77b4"          # 蓝色 (市场细分)
        price_level_base_color = "#ff7f0e" # 橙色 (价格水平标题)
        unit_price_color = "#2ca02c"        # 绿色 (均价预测)
        error_color = "#E74C3C"           # 红色 (通用错误或特定状态)
        success_color = "#2ECC71"         # 绿色 (特定状态)
        grey_color = "#7f7f7f"            # 灰色 (描述性文字或未知状态)

        # 使用列布局来并排显示三个预测结果
        col1, col2, col3 = st.columns(3)

        # --- 在列中显示结果 ---
        with col1: # 第一列：市场细分
            st.markdown(f"<h5 style='color: {market_color}; margin-bottom: 5px;'>市场细分</h5>", unsafe_allow_html=True)
            if market_pred_label != "预测失败" and market_pred_label != "处理中...":
                st.markdown(f"<p style='font-size: 28px; font-weight: bold; color: {market_color}; margin-bottom: 10px;'>{market_pred_label}</p>", unsafe_allow_html=True)
            else:
                st.error(f"预测失败") # 如果失败，显示错误提示
            st.markdown(f"<p style='font-size: small; color: {grey_color};'>判断房产在整体市场中的<br>价格定位。</p>", unsafe_allow_html=True)
            # --- 新增：显示使用的特征 ---
            with st.expander("查看使用特征"):
                if market_features_used:
                    st.caption(", ".join(market_features_used)) # 以逗号分隔显示特征列表
                else:
                    st.caption("未能获取特征列表。")

        with col2: # 第二列：价格水平
            st.markdown(f"<h5 style='color: {price_level_base_color}; margin-bottom: 5px;'>价格水平 (相对区域)</h5>", unsafe_allow_html=True)
            if price_level_pred_label != "预测失败" and price_level_pred_label != "处理中..." and price_level_pred_code != -99:
                # 根据预测编码设置显示颜色
                if price_level_pred_code == 1: display_text, display_color = price_level_pred_label, error_color # 高于区域均价 (红色)
                elif price_level_pred_code == 0: display_text, display_color = price_level_pred_label, success_color # 不高于区域均价 (绿色)
                else: display_text, display_color = "未知状态", grey_color # 其他情况 (灰色)
                st.markdown(f"<p style='font-size: 28px; font-weight: bold; color: {display_color}; margin-bottom: 10px;'>{display_text}</p>", unsafe_allow_html=True)
            else:
                st.error("预测失败")
            st.markdown(f"<p style='font-size: small; color: {grey_color};'>判断房产单价是否高于<br>其所在区域均值。</p>", unsafe_allow_html=True)
            # --- 新增：显示使用的特征 ---
            with st.expander("查看使用特征"):
                if price_level_features_used:
                    st.caption(", ".join(price_level_features_used))
                else:
                    st.caption("未能获取特征列表或预测失败。")

        with col3: # 第三列：均价预测
            st.markdown(f"<h5 style='color: {unit_price_color}; margin-bottom: 5px;'>均价预测</h5>", unsafe_allow_html=True)
            if regression_attempted and unit_price_pred != -1.0: # 检查是否尝试过且成功
                # 使用 Markdown 自定义样式，使数值颜色与标题一致
                st.markdown(f"""
                    <div style='margin-bottom: 10px;'>
                        <p style='font-size: small; color: {grey_color}; margin-bottom: 0px;'>预测单价 (元/㎡)</p>
                        <p style='font-size: 28px; font-weight: bold; color: {unit_price_color}; margin-top: 0px;'>
                            {unit_price_pred:,.0f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: small; color: {unit_price_color};'>预测的每平方米<br>大致价格。</p>", unsafe_allow_html=True)
            else: # 如果预测失败或未尝试
                st.error("预测失败")
                # 即使失败，也显示描述占位符
                st.markdown(f"<p style='font-size: small; color: {grey_color};'>预测的每平方米<br>大致价格。</p>", unsafe_allow_html=True)
            # --- 新增：显示使用的特征 ---
            with st.expander("查看使用特征"):
                 # 提示用户：这里的列表应不包含 '总价(万)' (如果模型训练时未使用)
                 st.info("提示：该预测通常不依赖'总价'输入。")
                 if regression_features_used:
                     st.caption(", ".join(regression_features_used))
                 else:
                     st.caption("未能获取特征列表或预测失败。")


        # --- 显示总体状态和错误信息 ---
        st.markdown("---") # 结果区域下方分隔线
        if not error_messages: # 如果没有错误信息
            st.success("✅ 分析预测完成！请查看上方结果。")
            st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
        else: # 如果有错误信息
             st.warning("⚠️ 部分或全部预测未能成功完成。")
             # 显示具体的错误信息列表
             st.error("执行过程中遇到的错误：")
             for i, msg in enumerate(error_messages):
                 st.markdown(f"{i+1}. {msg}") # 使用 markdown 列表显示错误

# --- 页脚信息 ---
st.sidebar.markdown("---") # 侧边栏分隔线
st.sidebar.caption("模型信息: LightGBM & RandomForest")
st.sidebar.caption("数据来源: 安居客") # 明确数据来源为模拟
st.sidebar.caption("开发者: 凌欢")