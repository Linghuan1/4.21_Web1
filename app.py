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
MARKET_MODEL_PATH = 'market_segment_lgbm_model.joblib' # 市场细分模型文件路径
PRICE_LEVEL_MODEL_PATH = 'price_level_rf_model.joblib' # 价格水平模型文件路径
REGRESSION_MODEL_PATH = 'unit_price_rf_model.joblib'   # 回归模型文件路径
SCALER_PATH = 'regression_scaler.joblib'             # 回归模型使用的 Scaler 文件路径
FEATURE_NAMES_PATH = 'feature_names.joblib'          # 特征名称列表文件路径
MAPPINGS_PATH = 'mappings.joblib'                    # 特征编码映射关系文件路径

# --- 加载资源函数 (使用缓存) ---
@st.cache_resource # Streamlit 缓存机制，避免每次交互都重新加载模型等资源
def load_resources():
    """加载所有必要的资源文件 (模型, scaler, 特征名, 映射关系)。"""
    resources = {} # 创建一个空字典来存储加载的资源
    all_files_exist = True # 标记所有文件是否存在
    required_files = [MARKET_MODEL_PATH, PRICE_LEVEL_MODEL_PATH, REGRESSION_MODEL_PATH,
                      SCALER_PATH, FEATURE_NAMES_PATH, MAPPINGS_PATH]

    # 检查所有文件是否存在于当前目录
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            # st.error(f"错误: 必需的资源文件未找到: {file_path}") # 不要在缓存函数内部使用st.error
            print(f"错误: 文件 {file_path} 未找到。")
            missing_files.append(file_path)
            all_files_exist = False

    if not all_files_exist:
        print(f"错误：缺少文件 {missing_files}。请确保所有 .joblib 文件与 app.py 在同一目录。")
        # 在函数外部处理 UI 错误信息
        return None, missing_files # 返回 None 和缺失的文件列表

    # 如果所有文件都存在，则尝试加载
    try:
        resources['market_model'] = joblib.load(MARKET_MODEL_PATH)
        resources['price_level_model'] = joblib.load(PRICE_LEVEL_MODEL_PATH)
        resources['regression_model'] = joblib.load(REGRESSION_MODEL_PATH)
        resources['scaler'] = joblib.load(SCALER_PATH)
        resources['feature_names'] = joblib.load(FEATURE_NAMES_PATH)
        # 加载映射文件
        resources['mappings'] = joblib.load(MAPPINGS_PATH) # <--- 直接使用 'mappings' 作为键
        print("所有资源加载成功。") # 在运行 Streamlit 的终端显示成功信息
        # 打印加载的映射以供调试
        print("从文件加载的映射关系:", resources['mappings'])
        # 打印加载的特征名称以供调试
        print("从文件加载的特征名称:", resources['feature_names'])
        return resources, None # 返回包含所有资源的字典和 None (表示没有缺失文件)
    except Exception as e:
        # 处理加载过程中可能出现的其他错误
        print(f"加载资源时发生错误: {e}")
        # 在函数外部处理 UI 错误信息
        return None, [f"加载错误: {e}"] # 返回 None 和错误信息

resources, load_error_info = load_resources() # 执行加载函数

# --- 辅助函数 ---
def format_mapping_options_for_selectbox(name_to_code_mapping):
    """
    为 Streamlit Selectbox 准备选项和格式化函数所需的数据。
    输入: 一个 '名称' -> '编码' 的字典 (例如 {'南北': 3, '东西': 4})
    输出: 一个 '编码' -> '显示字符串' 的字典 (例如 {3: '南北 (3)', 4: '东西 (4)'})
    """
    if not isinstance(name_to_code_mapping, dict):
        # st.error(f"格式化选项时出错：输入必须是字典，但收到了 {type(name_to_code_mapping)}") # 不要在辅助函数中用 st.error
        print(f"[格式化错误] 输入非字典: {type(name_to_code_mapping)}")
        return {} # 返回空字典以避免后续错误

    code_to_display_string = {}
    try:
        # 按编码值对 (名称, 编码) 对进行排序 (确保编码是可比较的，如整数)
        # 假设编码值已经是整数或可以转为整数
        sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        # 创建 编码 -> 显示字符串 的字典
        for name, code in sorted_items:
             # 确保 code 是整数，name 是字符串
            code_int = int(code)
            name_str = str(name)
            code_to_display_string[code_int] = f"{name_str} ({code_int})"
        return code_to_display_string
    except (ValueError, TypeError, KeyError) as e:
        # st.error(f"格式化选项时出错（检查映射关系）：{e}")
        print(f"[格式化错误] 转换/排序时出错: {e}")
        # 提供一个基本的回退格式
        return {int(v): f"{k} ({int(v)})" for k, v in name_to_code_mapping.items() if isinstance(v, (int, float, str)) and str(v).isdigit()}


# --- Streamlit 用户界面主要部分 ---
st.title("🏠 盐城二手房智能分析与预测") # 设置应用主标题
st.markdown("""
欢迎使用盐城二手房分析工具！请在左侧边栏输入房产特征，我们将为您提供三个维度的预测：
1.  **市场细分预测**: 判断房产属于低端、中端还是高端市场。
2.  **价格水平预测**: 判断房产单价是否高于其所在区域的平均水平。
3.  **房产均价预测**: 预测房产的每平方米单价（元/㎡）。
""") # 应用介绍文本
st.markdown("---") # 添加一条水平分隔线

# --- 应用启动时资源加载失败或映射缺失的处理 ---
if not resources:
     st.error("❌ **应用程序初始化失败！**")
     if load_error_info: # 如果有具体的错误信息
         st.warning(f"无法加载必要的资源文件。错误详情:")
         for error in load_error_info:
             st.markdown(f"*   `{error}`")
     else: # 如果只是文件找不到
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
        *   确保 `{MAPPINGS_PATH}` 文件包含所有必需的映射关系（方位、楼层、区域、房龄、市场类别、是否高于区域均价）。
        *   确保 `{FEATURE_NAMES_PATH}` 文件包含每个模型训练时使用的正确特征列表。
        *   检查运行 Streamlit 的终端是否有更详细的错误信息。
     """)
     st.stop() # 停止执行后续代码

# --- 如果资源加载成功，继续构建UI和预测逻辑 ---
# 从加载的资源中获取所需对象
mappings = resources['mappings']
feature_names = resources['feature_names']
market_model = resources['market_model']
price_level_model = resources['price_level_model']
regression_model = resources['regression_model']
scaler = resources['scaler']

# --- 检查 Mappings 和 Feature Names 是否符合预期 ---
required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价']
required_features = ['market', 'price_level', 'regression']
valid_resources = True

for key in required_mappings:
    if key not in mappings or not isinstance(mappings[key], dict):
        st.error(f"错误：加载的映射关系文件 `{MAPPINGS_PATH}` 中缺少或无效的键: '{key}'")
        valid_resources = False

for key in required_features:
    if key not in feature_names or not isinstance(feature_names[key], list):
        st.error(f"错误：加载的特征名称文件 `{FEATURE_NAMES_PATH}` 中缺少或无效的键: '{key}'")
        valid_resources = False

if not valid_resources:
    st.warning("资源文件内容不完整或格式错误，无法继续。请重新生成资源文件。")
    st.stop()

# --- 侧边栏输入控件 ---
st.sidebar.header("🏘️ 房产特征输入") # 侧边栏标题

# -- 分类型特征 (使用下拉选择框，基于加载的 mappings) --
st.sidebar.subheader("选择项特征")
selectbox_inputs = {} # 存储下拉框的选择结果 (编码值)

try:
    orientation_map = mappings['方位']
    orientation_display_map = format_mapping_options_for_selectbox(orientation_map)
    orientation_codes = list(orientation_display_map.keys())
    selectbox_inputs['方位'] = st.sidebar.selectbox(
        "房屋方位:", options=orientation_codes,
        format_func=lambda x: orientation_display_map.get(x, f"未知 ({x})"), key="orientation_select", help="选择房屋的主要朝向。"
    )
except Exception as e: st.sidebar.error(f"方位选项错误: {e}"); selectbox_inputs['方位'] = None

try:
    floor_map = mappings['楼层']
    floor_display_map = format_mapping_options_for_selectbox(floor_map)
    floor_codes = list(floor_display_map.keys())
    selectbox_inputs['楼层'] = st.sidebar.selectbox(
        "楼层类型:", options=floor_codes,
        format_func=lambda x: floor_display_map.get(x, f"未知 ({x})"), key="floor_select", help="选择房屋所在的楼层区间。"
    )
except Exception as e: st.sidebar.error(f"楼层选项错误: {e}"); selectbox_inputs['楼层'] = None

try:
    area_map = mappings['所属区域']
    area_display_map = format_mapping_options_for_selectbox(area_map)
    area_codes = list(area_display_map.keys())
    selectbox_inputs['所属区域'] = st.sidebar.selectbox(
        "所属区域:", options=area_codes,
        format_func=lambda x: area_display_map.get(x, f"未知 ({x})"), key="area_select", help="选择房产所在的行政区域。"
    )
except Exception as e: st.sidebar.error(f"区域选项错误: {e}"); selectbox_inputs['所属区域'] = None

try:
    age_map = mappings['房龄']
    age_display_map = format_mapping_options_for_selectbox(age_map)
    age_codes = list(age_display_map.keys())
    selectbox_inputs['房龄'] = st.sidebar.selectbox(
        "房龄:", options=age_codes,
        format_func=lambda x: age_display_map.get(x, f"未知 ({x})"), key="age_select", help="选择房屋的建造年限范围。"
    )
except Exception as e: st.sidebar.error(f"房龄选项错误: {e}"); selectbox_inputs['房龄'] = None


# -- 数值型特征 (使用数字输入框) --
st.sidebar.subheader("数值项特征")
numeric_inputs = {} # 存储数值输入结果

numeric_inputs['总价(万)'] = st.sidebar.number_input(
    "总价 (万):", min_value=10.0, max_value=1500.0, value=100.0, step=5.0, format="%.1f", key="total_price", help="输入房产的总价，单位万元。"
)
numeric_inputs['面积(㎡)'] = st.sidebar.number_input(
    "面积 (㎡):", min_value=30.0, max_value=600.0, value=100.0, step=5.0, format="%.1f", key="area_sqm", help="输入房产的建筑面积，单位平方米。"
)
numeric_inputs['建造时间'] = st.sidebar.number_input(
    "建造时间 (年份):", min_value=1970, max_value=2025, value=2018, step=1, format="%d", key="build_year", help="输入房屋的建造年份。"
)
# 在 'save_models.py' 中 '楼层数' 是 'floor_num', Streamlit 中是 '总楼层数:'，
# 需要确认训练时用的特征名是哪个。假设是 '楼层数'
numeric_inputs['楼层数'] = st.sidebar.number_input(
    "总楼层数:", min_value=1, max_value=60, value=18, step=1, format="%d", key="floor_num", help="输入楼栋的总楼层数。"
)
numeric_inputs['室'] = st.sidebar.number_input(
    "室:", min_value=1, max_value=10, value=3, step=1, format="%d", key="rooms", help="输入卧室数量。"
)
numeric_inputs['厅'] = st.sidebar.number_input(
    "厅:", min_value=0, max_value=5, value=2, step=1, format="%d", key="halls", help="输入客厅/餐厅数量。"
)
numeric_inputs['卫'] = st.sidebar.number_input(
    "卫:", min_value=0, max_value=5, value=1, step=1, format="%d", key="baths", help="输入卫生间数量。"
)

# --- 预测触发按钮 ---
st.sidebar.markdown("---") # 侧边栏分隔线
if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help="点击这里根据输入的特征进行预测分析"):

    # 检查所有必需的 selectbox 是否都有有效值 (不是 None)
    if None in selectbox_inputs.values():
        st.error("⚠️ **输入错误：** 请确保所有下拉选择框（方位、楼层、区域、房龄）都有有效的选项。如果选项加载失败，请检查代码或相关错误提示。")
    else:
        # --- 准备三个模型各自所需的输入 DataFrame ---
        prediction_possible = True # 标记预测是否可以进行
        error_messages = [] # 收集错误信息
        market_pred_label = "未进行预测"
        price_level_pred_label = "未进行预测"
        price_level_pred_code = -1 # 初始化
        unit_price_pred = -1 # 初始化

        # 合并所有输入到一个字典，方便按需取用
        all_inputs = {**selectbox_inputs, **numeric_inputs}

        # 1. 为市场细分模型准备输入
        try:
            market_features_needed = feature_names['market']
            # 从 all_inputs 中提取市场细分模型需要的特征
            input_data_market = {feat: all_inputs[feat] for feat in market_features_needed if feat in all_inputs}
            # 检查是否所有需要的特征都已提取
            if len(input_data_market) != len(market_features_needed):
                missing = set(market_features_needed) - set(input_data_market.keys())
                raise KeyError(f"市场细分模型缺少输入特征: {missing}")

            input_df_market = pd.DataFrame([input_data_market])
            input_df_market = input_df_market[market_features_needed] # 确保特征顺序严格一致
            market_pred_code = market_model.predict(input_df_market)[0]
            market_output_map = mappings.get('市场类别', {})
            market_pred_label = market_output_map.get(int(market_pred_code), f"预测编码无效 ({market_pred_code})")
            print(f"市场细分输入: {input_df_market.to_dict()}") # 调试打印
            print(f"市场细分预测代码: {market_pred_code}, 标签: {market_pred_label}")
        except KeyError as e:
            msg = f"市场细分模型输入准备错误: {e}。请检查 feature_names.joblib 文件和侧边栏输入。"
            st.error(msg); print(msg); error_messages.append(msg); prediction_possible = False
        except Exception as e:
            msg = f"市场细分模型预测时发生错误: {e}"
            st.error(msg); print(msg); error_messages.append(msg); prediction_possible = False

        # 2. 为价格水平模型准备输入 (仅当之前步骤成功)
        if prediction_possible:
            try:
                price_level_features_needed = feature_names['price_level']
                input_data_price_level = {feat: all_inputs[feat] for feat in price_level_features_needed if feat in all_inputs}
                if len(input_data_price_level) != len(price_level_features_needed):
                     missing = set(price_level_features_needed) - set(input_data_price_level.keys())
                     raise KeyError(f"价格水平模型缺少输入特征: {missing}")

                input_df_price_level = pd.DataFrame([input_data_price_level])
                input_df_price_level = input_df_price_level[price_level_features_needed] # 确保顺序
                price_level_pred_code = price_level_model.predict(input_df_price_level)[0]
                price_level_output_map = mappings.get('是否高于区域均价', {})
                price_level_pred_label = price_level_output_map.get(int(price_level_pred_code), f"预测编码无效 ({price_level_pred_code})")
                print(f"价格水平输入: {input_df_price_level.to_dict()}") # 调试打印
                print(f"价格水平预测代码: {price_level_pred_code}, 标签: {price_level_pred_label}")
            except KeyError as e:
                msg = f"价格水平模型输入准备错误: {e}。请检查 feature_names.joblib 文件和侧边栏输入。"
                st.error(msg); print(msg); error_messages.append(msg); prediction_possible = False
            except Exception as e:
                msg = f"价格水平模型预测时发生错误: {e}"
                st.error(msg); print(msg); error_messages.append(msg); prediction_possible = False

        # 3. 为回归模型准备输入 (仅当之前步骤成功)
        if prediction_possible:
            try:
                regression_features_needed = feature_names['regression']
                input_data_reg = {feat: all_inputs[feat] for feat in regression_features_needed if feat in all_inputs}
                if len(input_data_reg) != len(regression_features_needed):
                     missing = set(regression_features_needed) - set(input_data_reg.keys())
                     raise KeyError(f"均价预测模型缺少输入特征: {missing}")

                input_df_reg = pd.DataFrame([input_data_reg])
                input_df_reg = input_df_reg[regression_features_needed] # 确保顺序
                input_df_reg_scaled = scaler.transform(input_df_reg) # 标准化
                unit_price_pred = regression_model.predict(input_df_reg_scaled)[0]
                unit_price_pred = max(0, unit_price_pred) # 确保价格不为负
                print(f"回归模型输入 (原始): {input_df_reg.to_dict()}") # 调试打印
                print(f"回归模型输入 (缩放后): {input_df_reg_scaled}")
                print(f"回归模型预测值: {unit_price_pred}")
            except KeyError as e:
                msg = f"均价预测模型输入准备错误: {e}。请检查 feature_names.joblib 文件和侧边栏输入。"
                st.error(msg); print(msg); error_messages.append(msg); unit_price_pred = -1; prediction_possible = False # 即使出错也标记为-1
            except Exception as e:
                msg = f"均价预测模型预测时发生错误: {e}"
                st.error(msg); print(msg); error_messages.append(msg); unit_price_pred = -1; prediction_possible = False # 即使出错也标记为-1

        # --- 在主页面分列显示预测结果 ---
        # 只有在没有收集到错误信息时才显示完整结果区
        if not error_messages:
            st.markdown("---") # 分隔线
            st.subheader("📈 预测结果分析") # 结果区域的总标题
            col1, col2, col3 = st.columns(3) # 创建三列布局

            with col1: # 市场细分
                st.markdown("<h5 style='text-align: center; color: #1f77b4; margin-bottom: 0px;'>市场细分</h5>", unsafe_allow_html=True)
                st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                # 检查标签是否包含错误提示词
                if "错误" not in market_pred_label and "无效" not in market_pred_label and market_pred_label != "未进行预测":
                    st.markdown(f"<p style='text-align: center; font-size: 28px; font-weight: bold; color: #1f77b4;'>{market_pred_label}</p>", unsafe_allow_html=True)
                    st.caption("判断房产在整体市场中的价格定位。")
                else:
                    st.warning(f"市场细分预测失败: {market_pred_label}")

            with col2: # 价格水平
                st.markdown("<h5 style='text-align: center; color: #ff7f0e; margin-bottom: 0px;'>价格水平 (相对区域)</h5>", unsafe_allow_html=True)
                st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                if "错误" not in price_level_pred_label and "无效" not in price_level_pred_label and price_level_pred_label != "未进行预测":
                     # 假设编码 1 代表 '是 (高于)', 0 代表 '否 (不高于)'
                     if price_level_pred_code == 1: display_text, display_color = price_level_pred_label, "#E74C3C" # 红色表示高于
                     elif price_level_pred_code == 0: display_text, display_color = price_level_pred_label, "#2ECC71" # 绿色表示不高于
                     else: display_text, display_color = "未知状态", "#7f7f7f" # 灰色表示未知
                     st.markdown(f"<p style='text-align: center; font-size: 28px; font-weight: bold; color: {display_color};'>{display_text}</p>", unsafe_allow_html=True)
                     st.caption("判断房产单价是否高于其所在区域均值。")
                else:
                     st.warning(f"价格水平预测失败: {price_level_pred_label}")

            with col3: # 均价预测
                st.markdown("<h5 style='text-align: center; color: #2ca02c; margin-bottom: 0px;'>均价预测</h5>", unsafe_allow_html=True)
                st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                if unit_price_pred != -1: # 检查是否成功预测
                     st.metric(label="预测单价 (元/㎡)", value=f"{unit_price_pred:,.0f}") # 使用千位分隔符
                     st.caption("预测的每平方米大致价格。")
                else:
                     st.warning("无法完成房产均价预测。")

            st.success("✅ 分析预测完成！请查看上方结果。") # 显示一个成功的提示消息
            st.markdown("---") # 结束分隔线
            st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
        else:
            # 如果有错误信息，在这里统一提示
            st.warning("部分或全部预测因输入或模型准备错误未能完成，请检查上方错误信息。")


# --- 页脚信息 ---
st.sidebar.markdown("---") # 侧边栏分隔线
st.sidebar.caption("模型信息: LightGBM & RandomForest") # 模型信息
st.sidebar.caption("数据来源: 安居客") # 数据来源说明，改为示例更准确
st.sidebar.caption("开发者: 凌欢") # 开发者信息