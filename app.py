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

# --- 常量定义：模型和资源文件路径 (直接在当前目录下) ---
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
    for file_path in required_files:
        if not os.path.exists(file_path):
            st.error(f"错误: 必需的资源文件未找到: {file_path}")
            print(f"错误: 文件 {file_path} 未找到。")
            all_files_exist = False

    if not all_files_exist:
        # 更新错误提示，说明文件应在同一目录下
        st.error("请确保所有必需的 .joblib 文件与 app.py 文件在同一个目录下。")
        return None # 如果文件缺失，则返回 None

    # 如果所有文件都存在，则尝试加载
    try:
        resources['market_model'] = joblib.load(MARKET_MODEL_PATH)
        resources['price_level_model'] = joblib.load(PRICE_LEVEL_MODEL_PATH)
        resources['regression_model'] = joblib.load(REGRESSION_MODEL_PATH)
        resources['scaler'] = joblib.load(SCALER_PATH)
        resources['feature_names'] = joblib.load(FEATURE_NAMES_PATH)
        # 加载映射文件
        resources['loaded_mappings'] = joblib.load(MAPPINGS_PATH) # <--- 加载原始文件内容
        print("所有资源加载成功。") # 在运行 Streamlit 的终端显示成功信息
        # 打印加载的映射以供调试
        print("从文件加载的原始映射关系:", resources['loaded_mappings'])
        return resources # 返回包含所有资源的字典
    except Exception as e:
        # 处理加载过程中可能出现的其他错误
        st.error(f"加载资源时发生错误: {e}")
        print(f"加载资源时发生错误: {e}")
        return None # 返回 None 表示加载失败

resources = load_resources() # 执行加载函数

# --- 辅助函数 ---
def format_mapping_options_for_selectbox(name_to_code_mapping):
    """
    为 Streamlit Selectbox 准备选项和格式化函数所需的数据。
    输入: 一个 '名称' -> '编码' 的字典 (例如 {'南北': 3, '东西': 4})
    输出: 一个 '编码' -> '显示字符串' 的字典 (例如 {3: '南北 (3)', 4: '东西 (4)'})
    """
    if not isinstance(name_to_code_mapping, dict):
        st.error(f"格式化选项时出错：输入必须是字典，但收到了 {type(name_to_code_mapping)}")
        return {} # 返回空字典以避免后续错误

    code_to_display_string = {}
    try:
        # 按编码值对 (名称, 编码) 对进行排序
        sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        # 创建 编码 -> 显示字符串 的字典
        for name, code in sorted_items:
             # 确保 code 是整数，name 是字符串
            code_int = int(code)
            name_str = str(name)
            code_to_display_string[code_int] = f"{name_str} ({code_int})"
        return code_to_display_string
    except (ValueError, TypeError, KeyError) as e:
        st.error(f"格式化选项时出错（检查代码中定义的 correct_mappings）：{e}")
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

if resources: # 只有当所有资源成功加载时，才继续构建UI和执行预测逻辑
    # loaded_mappings = resources['loaded_mappings'] # 获取从文件加载的原始映射
    feature_names = resources['feature_names'] # 获取加载的特征名称列表

    # --- 在代码中直接定义正确的映射关系 ---
    # 这些将覆盖从 mappings.joblib 加载的错误映射 (用于 Selectbox)
    correct_mappings = {
        '方位': {'东': 0, '北': 1, '南': 2, '南北': 3, '西': 4, '西北': 5, '西南': 6},
        '楼层': {'中层': 0, '低层': 1, '高层': 2},
        '所属区域': {'东台': 0, '亭湖': 1, '响水': 2, '大丰': 3, '射阳': 4, '建湖': 5, '滨海': 6, '盐都': 7, '阜宁': 8},
        '房龄': {'2-5年': 0, '2年内': 1}
    }
    # 从加载的文件中获取我们仍然需要的映射
    try:
        loaded_mappings_used = {
            '市场类别': resources['loaded_mappings']['市场类别'],
            '是否高于区域均价': resources['loaded_mappings']['是否高于区域均价']
        }
        # 验证这些加载的映射是否看起来合理
        if not isinstance(loaded_mappings_used['市场类别'], dict) or not isinstance(loaded_mappings_used['是否高于区域均价'], dict):
            raise ValueError("从文件加载的 '市场类别' 或 '是否高于区域均价' 映射格式不正确。")
        print("成功提取 '市场类别' 和 '是否高于区域均价' 的映射。")
    except KeyError as e:
        st.error(f"错误：无法从 `mappings.joblib` 文件中找到 '{e}' 的映射关系。这个映射是必需的。")
        # 如果关键映射缺失，则无法继续
        resources = None # 将 resources 设为 None 以阻止后续 UI 渲染
    except ValueError as e:
        st.error(f"错误：{e}")
        resources = None
    except Exception as e:
        st.error(f"处理加载的映射时发生未知错误: {e}")
        resources = None

# 重新检查 resources 是否仍然有效
if resources:
    # --- 侧边栏输入控件 ---
    st.sidebar.header("🏘️ 房产特征输入") # 侧边栏标题

    # -- 分类型特征 (使用下拉选择框，基于 correct_mappings) --
    st.sidebar.subheader("选择项特征")

    # 使用 correct_mappings 设置 Selectbox
    try:
        orientation_display_map = format_mapping_options_for_selectbox(correct_mappings['方位'])
        orientation_codes = list(orientation_display_map.keys())
        selected_orientation = st.sidebar.selectbox(
            "房屋方位:", options=orientation_codes,
            format_func=lambda x: orientation_display_map.get(x, f"未知 ({x})"), key="orientation_select", help="选择房屋的主要朝向。"
        )
    except Exception as e: st.sidebar.error(f"方位选项错误: {e}"); selected_orientation = None

    try:
        floor_display_map = format_mapping_options_for_selectbox(correct_mappings['楼层'])
        floor_codes = list(floor_display_map.keys())
        selected_floor = st.sidebar.selectbox(
            "楼层类型:", options=floor_codes,
            format_func=lambda x: floor_display_map.get(x, f"未知 ({x})"), key="floor_select", help="选择房屋所在的楼层区间。"
        )
    except Exception as e: st.sidebar.error(f"楼层选项错误: {e}"); selected_floor = None

    try:
        area_display_map = format_mapping_options_for_selectbox(correct_mappings['所属区域'])
        area_codes = list(area_display_map.keys())
        selected_area = st.sidebar.selectbox(
            "所属区域:", options=area_codes,
            format_func=lambda x: area_display_map.get(x, f"未知 ({x})"), key="area_select", help="选择房产所在的行政区域。"
        )
    except Exception as e: st.sidebar.error(f"区域选项错误: {e}"); selected_area = None

    try:
        age_display_map = format_mapping_options_for_selectbox(correct_mappings['房龄'])
        age_codes = list(age_display_map.keys())
        selected_age = st.sidebar.selectbox(
            "房龄:", options=age_codes,
            format_func=lambda x: age_display_map.get(x, f"未知 ({x})"), key="age_select", help="选择房屋的建造年限范围。"
        )
    except Exception as e: st.sidebar.error(f"房龄选项错误: {e}"); selected_age = None

    # -- 数值型特征 (使用数字输入框) --
    st.sidebar.subheader("数值项特征")
    total_price = st.sidebar.number_input(
        "总价 (万):", min_value=10.0, max_value=1500.0, value=100.0, step=5.0, format="%.1f", help="输入房产的总价，单位万元。"
    )
    area_sqm = st.sidebar.number_input(
        "面积 (㎡):", min_value=30.0, max_value=600.0, value=100.0, step=5.0, format="%.1f", help="输入房产的建筑面积，单位平方米。"
    )
    build_year = st.sidebar.number_input(
        "建造时间 (年份):", min_value=1970, max_value=2025, value=2018, step=1, format="%d", help="输入房屋的建造年份。"
    )
    floor_num = st.sidebar.number_input(
        "总楼层数:", min_value=1, max_value=60, value=18, step=1, format="%d", help="输入楼栋的总楼层数。"
    )
    rooms = st.sidebar.number_input(
        "室:", min_value=1, max_value=10, value=3, step=1, format="%d", help="输入卧室数量。"
    )
    halls = st.sidebar.number_input(
        "厅:", min_value=0, max_value=5, value=2, step=1, format="%d", help="输入客厅/餐厅数量。"
    )
    baths = st.sidebar.number_input(
        "卫:", min_value=0, max_value=5, value=1, step=1, format="%d", help="输入卫生间数量。"
    )

    # --- 预测触发按钮 ---
    st.sidebar.markdown("---") # 侧边栏分隔线
    if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help="点击这里根据输入的特征进行预测分析"):

        # 检查所有必需的 selectbox 是否都有有效值 (不是 None)
        if None in [selected_orientation, selected_floor, selected_area, selected_age]:
            st.error("⚠️ **输入错误：** 请确保所有下拉选择框（方位、楼层、区域、房龄）都有有效的选项。如果选项加载失败，请检查代码或相关错误提示。")
        else:
            # --- 准备三个模型各自所需的输入 DataFrame ---
            prediction_possible = True # 标记预测是否可以进行
            market_pred_label = "未进行预测"
            price_level_pred_label = "未进行预测"
            price_level_pred_code = -1 # 初始化
            unit_price_pred = -1 # 初始化

            # 1. 为市场细分模型准备输入
            try:
                input_data_market = {
                    '方位': selected_orientation, '楼层': selected_floor, '所属区域': selected_area,
                    '房龄': selected_age, '建造时间': build_year, '楼层数': floor_num,
                    '室': rooms, '厅': halls, '卫': baths
                }
                input_df_market = pd.DataFrame([input_data_market])
                input_df_market = input_df_market[feature_names['market']] # 确保特征顺序
                market_pred_code = resources['market_model'].predict(input_df_market)[0]
                # !! 使用从文件加载的 loaded_mappings_used 来获取输出标签 !!
                market_pred_label = loaded_mappings_used['市场类别'].get(int(market_pred_code), f"预测编码无效 ({market_pred_code})")
            except KeyError as e: st.error(f"市场细分模型输入准备错误: 特征 '{e}'。"); market_pred_label = "错误"; prediction_possible = False
            except Exception as e: st.error(f"市场细分模型预测出错: {e}"); market_pred_label = "错误"; prediction_possible = False

            # 2. 为价格水平模型准备输入
            if prediction_possible:
                try:
                    input_data_price_level = {
                        '总价(万)': total_price, '所属区域': selected_area, '建造时间': build_year,
                        '楼层数': floor_num, '面积(㎡)': area_sqm
                    }
                    input_df_price_level = pd.DataFrame([input_data_price_level])
                    input_df_price_level = input_df_price_level[feature_names['price_level']] # 确保特征顺序
                    price_level_pred_code = resources['price_level_model'].predict(input_df_price_level)[0]
                     # !! 使用从文件加载的 loaded_mappings_used 来获取输出标签 !!
                    price_level_pred_label = loaded_mappings_used['是否高于区域均价'].get(int(price_level_pred_code), f"预测编码无效 ({price_level_pred_code})")
                except KeyError as e: st.error(f"价格水平模型输入准备错误: 特征 '{e}'。"); price_level_pred_label = "错误"; prediction_possible = False
                except Exception as e: st.error(f"价格水平模型预测出错: {e}"); price_level_pred_label = "错误"; prediction_possible = False

            # 3. 为回归模型准备输入
            if prediction_possible:
                try:
                    input_data_reg = {
                        '总价(万)': total_price, '所属区域': selected_area, '建造时间': build_year,
                        '楼层数': floor_num, '房龄': selected_age, '室': rooms, '厅': halls, '卫': baths
                    }
                    input_df_reg = pd.DataFrame([input_data_reg])
                    input_df_reg = input_df_reg[feature_names['regression']] # 确保特征顺序
                    input_df_reg_scaled = resources['scaler'].transform(input_df_reg) # 标准化
                    unit_price_pred = resources['regression_model'].predict(input_df_reg_scaled)[0]
                    unit_price_pred = max(0, unit_price_pred) # 确保价格不为负
                except KeyError as e: st.error(f"均价预测模型输入准备错误: 特征 '{e}'。"); unit_price_pred = -1; prediction_possible = False
                except Exception as e: st.error(f"均价预测模型预测出错: {e}"); unit_price_pred = -1; prediction_possible = False

            # --- 在主页面分列显示预测结果 ---
            if prediction_possible: # 只有在所有步骤都可能执行时才显示结果区
                st.markdown("---") # 分隔线
                st.subheader("📈 预测结果分析") # 结果区域的总标题
                col1, col2, col3 = st.columns(3) # 创建三列布局

                with col1: # 市场细分
                    st.markdown("<h5 style='text-align: center; color: #1f77b4; margin-bottom: 0px;'>市场细分</h5>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    if "错误" not in market_pred_label and "无效" not in market_pred_label:
                        st.markdown(f"<p style='text-align: center; font-size: 28px; font-weight: bold; color: #1f77b4;'>{market_pred_label}</p>", unsafe_allow_html=True)
                        st.caption("判断房产在整体市场中的价格定位。")
                    else: st.error(market_pred_label)

                with col2: # 价格水平
                    st.markdown("<h5 style='text-align: center; color: #ff7f0e; margin-bottom: 0px;'>价格水平 (相对区域)</h5>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    if "错误" not in price_level_pred_label and "无效" not in price_level_pred_label:
                         if price_level_pred_code == 1: display_text, display_color = price_level_pred_label, "#E74C3C" # 高于
                         else: display_text, display_color = price_level_pred_label, "#2ECC71" # 不高于
                         st.markdown(f"<p style='text-align: center; font-size: 28px; font-weight: bold; color: {display_color};'>{display_text}</p>", unsafe_allow_html=True)
                         st.caption("判断房产单价是否高于其所在区域均值。")
                    else: st.error(price_level_pred_label)

                with col3: # 均价预测
                    st.markdown("<h5 style='text-align: center; color: #2ca02c; margin-bottom: 0px;'>均价预测</h5>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    if unit_price_pred != -1:
                         st.metric(label="预测单价 (元/㎡)", value=f"{unit_price_pred:,.0f}")
                         st.caption("预测的每平方米大致价格。")
                    else: st.error("无法完成房产均价预测。")

                st.success("✅ 分析预测完成！请查看上方结果。") # 显示一个成功的提示消息
                st.markdown("---") # 结束分隔线
                st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
            else:
                st.warning("部分或全部预测因输入或模型准备错误未能完成，请检查上方错误信息。")


# --- 应用启动时资源加载失败或关键映射缺失的处理 ---
elif not resources:
     # 如果 load_resources() 返回 None，说明加载失败或关键映射缺失
     st.error("❌ 应用程序初始化失败！")
     st.warning("无法加载必要的模型或数据文件，或文件内容不完整。请检查以下几点：")
     st.markdown(f"""
        *   确认以下所有 `.joblib` 文件都与 `app.py` 文件在 **同一个** 目录下:
            *   `{MARKET_MODEL_PATH}`
            *   `{PRICE_LEVEL_MODEL_PATH}`
            *   `{REGRESSION_MODEL_PATH}`
            *   `{SCALER_PATH}`
            *   `{FEATURE_NAMES_PATH}`
            *   `{MAPPINGS_PATH}`
        *   特别检查 `{MAPPINGS_PATH}` 文件是否包含有效的 '市场类别' 和 '是否高于区域均价' 映射。
        *   检查运行 Streamlit 的终端是否有更详细的错误信息。
     """)

# --- 页脚信息 ---
st.sidebar.markdown("---") # 侧边栏分隔线
st.sidebar.caption("模型信息: LightGBM & RandomForest") # 模型信息
st.sidebar.caption("数据来源: 安居客 ") # 数据来源说明
st.sidebar.caption("开发者: [凌欢]")