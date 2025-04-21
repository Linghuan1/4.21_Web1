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
# MODEL_DIR = "saved_models" # 不再需要模型目录变量
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
        resources['mappings'] = joblib.load(MAPPINGS_PATH)
        print("所有资源加载成功。") # 在运行 Streamlit 的终端显示成功信息
        return resources # 返回包含所有资源的字典
    except Exception as e:
        # 处理加载过程中可能出现的其他错误
        st.error(f"加载资源时发生错误: {e}")
        print(f"加载资源时发生错误: {e}")
        return None # 返回 None 表示加载失败

resources = load_resources() # 执行加载函数

# --- 辅助函数 ---
def format_mapping_options(mapping_dict):
    """格式化编码映射字典，用于 Streamlit Selectbox 的选项显示。
       格式为: 显示名称 (编码值)。
    """
    # 按编码值（字典的值）对项进行排序，然后创建新的显示字典
    # 确保编码值是整数以便排序
    try:
        return {int(code): f"{name} ({int(code)})"
                for name, code in sorted(mapping_dict.items(), key=lambda item: int(item[1]))}
    except (ValueError, TypeError) as e:
        st.error(f"处理映射关系时出错：编码值必须是数字。错误：{e}")
        # 提供一个回退，即使排序/格式化失败也能显示点东西
        return {code: f"{name} ({code})" for name, code in mapping_dict.items()}


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
    mappings = resources['mappings'] # 获取加载的映射关系
    feature_names = resources['feature_names'] # 获取加载的特征名称列表

    # --- 侧边栏输入控件 ---
    st.sidebar.header("🏘️ 房产特征输入") # 侧边栏标题

    # -- 分类型特征 (使用下拉选择框) --
    st.sidebar.subheader("选择项特征")
    # 为每个分类特征准备格式化后的选项，用于 selectbox 显示
    # 使用 try-except 块增加健壮性，防止某个映射不存在导致整个应用崩溃
    try:
        orientation_options = format_mapping_options(mappings['方位'])
        selected_orientation = st.sidebar.selectbox(
            "房屋方位:",
            options=list(orientation_options.keys()),  # 选项是编码值列表
            format_func=lambda x: orientation_options[x], # 显示格式化后的文本
            key="orientation_select", # 添加唯一 key
            help="选择房屋的主要朝向。" # 添加提示信息
        )
    except KeyError:
        st.sidebar.error("错误：缺少 '方位' 的映射关系。")
        selected_orientation = None # 设置为 None 或默认值
    except Exception as e:
        st.sidebar.error(f"加载 '方位' 选项时出错: {e}")
        selected_orientation = None

    try:
        floor_options = format_mapping_options(mappings['楼层'])
        selected_floor = st.sidebar.selectbox(
            "楼层类型:",
            options=list(floor_options.keys()),
            format_func=lambda x: floor_options[x],
            key="floor_select",
            help="选择房屋所在的楼层区间（高、中、低等）。"
        )
    except KeyError:
        st.sidebar.error("错误：缺少 '楼层' 的映射关系。")
        selected_floor = None
    except Exception as e:
        st.sidebar.error(f"加载 '楼层' 选项时出错: {e}")
        selected_floor = None

    try:
        area_options = format_mapping_options(mappings['所属区域'])
        selected_area = st.sidebar.selectbox(
            "所属区域:",
            options=list(area_options.keys()),
            format_func=lambda x: area_options[x],
            key="area_select",
            help="选择房产所在的行政区域或板块。"
        )
    except KeyError:
        st.sidebar.error("错误：缺少 '所属区域' 的映射关系。")
        selected_area = None
    except Exception as e:
        st.sidebar.error(f"加载 '所属区域' 选项时出错: {e}")
        selected_area = None

    try:
        age_options = format_mapping_options(mappings['房龄'])
        selected_age = st.sidebar.selectbox(
            "房龄:",
            options=list(age_options.keys()),
            format_func=lambda x: age_options[x],
            key="age_select",
            help="选择房屋的建造年限范围。"
        )
    except KeyError:
        st.sidebar.error("错误：缺少 '房龄' 的映射关系。")
        selected_age = None
    except Exception as e:
        st.sidebar.error(f"加载 '房龄' 选项时出错: {e}")
        selected_age = None

    # -- 数值型特征 (使用数字输入框) --
    st.sidebar.subheader("数值项特征")
    # 创建数字输入框，允许用户输入或调整数值
    total_price = st.sidebar.number_input(
        "总价 (万):",
        min_value=10.0, max_value=1500.0, value=100.0, step=5.0, format="%.1f",
        help="输入房产的总价，单位为万元。"
    )
    area_sqm = st.sidebar.number_input(
        "面积 (㎡):",
        min_value=30.0, max_value=600.0, value=100.0, step=5.0, format="%.1f",
        help="输入房产的建筑面积，单位为平方米。"
    )
    build_year = st.sidebar.number_input(
        "建造时间 (年份):",
        min_value=1970, max_value=2025, value=2018, step=1, format="%d",
        help="输入房屋的建造年份。"
    )
    floor_num = st.sidebar.number_input(
        "总楼层数:",
        min_value=1, max_value=60, value=18, step=1, format="%d",
        help="输入楼栋的总楼层数。"
    )
    rooms = st.sidebar.number_input(
        "室:",
        min_value=1, max_value=10, value=3, step=1, format="%d",
        help="输入卧室的数量。"
    )
    halls = st.sidebar.number_input(
        "厅:",
        min_value=0, max_value=5, value=2, step=1, format="%d",
        help="输入客厅/餐厅的数量。"
    )
    baths = st.sidebar.number_input(
        "卫:",
        min_value=0, max_value=5, value=1, step=1, format="%d",
        help="输入卫生间的数量。"
    )

    # --- 预测触发按钮 ---
    st.sidebar.markdown("---") # 侧边栏分隔线
    # 创建一个按钮，点击后触后端的预测逻辑
    if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help="点击这里根据输入的特征进行预测分析"):

        # 在执行预测前，检查所有必需的 selectbox 是否都有有效值
        if None in [selected_orientation, selected_floor, selected_area, selected_age]:
            st.error("错误：请先确保所有下拉选择框都有有效的选项，并修复映射关系错误（如果存在）。")
        else:
            # --- 准备三个模型各自所需的输入 DataFrame ---
            prediction_possible = True # 标记预测是否可以进行
            market_pred_label = "未进行预测"
            price_level_pred_label = "未进行预测"
            price_level_pred_code = -1 # 初始化
            unit_price_pred = -1 # 初始化

            # 1. 为市场细分模型准备输入
            try:
                # 注意：这里的特征需要与 market_model 训练时完全一致
                input_data_market = {
                    '方位': selected_orientation, '楼层': selected_floor, '所属区域': selected_area,
                    '房龄': selected_age, '建造时间': build_year, '楼层数': floor_num,
                    '室': rooms, '厅': halls, '卫': baths
                }
                input_df_market = pd.DataFrame([input_data_market])
                # 严格按照训练时的特征顺序排列 DataFrame 的列
                input_df_market = input_df_market[feature_names['market']]
                # 使用加载的模型进行预测
                market_pred_code = resources['market_model'].predict(input_df_market)[0]
                # 从映射关系中查找预测编码对应的标签文本
                market_pred_label = mappings['市场类别'].get(int(market_pred_code), f"预测编码无效 ({market_pred_code})")
            except KeyError as e:
                 st.error(f"市场细分模型输入准备错误: 缺少或无法匹配特征 '{e}'。请检查 feature_names.joblib 文件。")
                 market_pred_label = "错误: 输入特征不匹配"
                 prediction_possible = False
            except Exception as e:
                 st.error(f"市场细分模型预测时发生意外错误: {e}")
                 market_pred_label = "错误: 预测失败"
                 prediction_possible = False

            # 2. 为价格水平模型准备输入 (仅当上一步没有关键错误时进行)
            if prediction_possible:
                try:
                    # 注意：这里的特征需要与 price_level_model 训练时完全一致
                    input_data_price_level = {
                        '总价(万)': total_price,
                        '所属区域': selected_area,
                        '建造时间': build_year,
                        '楼层数': floor_num,
                        '面积(㎡)': area_sqm # 确认这个特征是否真的用于此模型
                    }
                    input_df_price_level = pd.DataFrame([input_data_price_level])
                    # 严格按照训练时的特征顺序排列 DataFrame 的列
                    input_df_price_level = input_df_price_level[feature_names['price_level']]
                    # 使用加载的模型进行预测
                    price_level_pred_code = resources['price_level_model'].predict(input_df_price_level)[0]
                    price_level_pred_label = mappings['是否高于区域均价'].get(int(price_level_pred_code), f"预测编码无效 ({price_level_pred_code})")
                except KeyError as e:
                     st.error(f"价格水平模型输入准备错误: 缺少或无法匹配特征 '{e}'。请检查 feature_names.joblib 文件中 'price_level' 的特征列表。")
                     price_level_pred_label = "错误: 输入特征不匹配"
                     prediction_possible = False
                except Exception as e:
                     st.error(f"价格水平模型预测时发生意外错误: {e}")
                     price_level_pred_label = "错误: 预测失败"
                     prediction_possible = False

            # 3. 为回归模型准备输入 (仅当之前没有关键错误时进行)
            if prediction_possible:
                try:
                    # 注意：这里的特征需要与 regression_model 训练时完全一致
                    input_data_reg = {
                        '总价(万)': total_price,
                        '所属区域': selected_area,
                        '建造时间': build_year,
                        '楼层数': floor_num,
                        '房龄': selected_age,
                        '室': rooms,
                        '厅': halls,
                        '卫': baths
                    }
                    input_df_reg = pd.DataFrame([input_data_reg])
                    # 严格按照训练时的特征顺序排列 DataFrame 的列
                    input_df_reg = input_df_reg[feature_names['regression']]
                    # !!! 重要：使用加载的 scaler 对输入数据进行标准化 !!!
                    input_df_reg_scaled = resources['scaler'].transform(input_df_reg)
                    # 使用加载的回归模型进行预测
                    unit_price_pred = resources['regression_model'].predict(input_df_reg_scaled)[0]
                     # 对预测结果进行合理性约束，例如，价格不能为负数
                    unit_price_pred = max(0, unit_price_pred) # 确保价格不为负
                except KeyError as e:
                     st.error(f"均价预测模型输入准备错误: 缺少或无法匹配特征 '{e}'。请检查 feature_names.joblib 文件中 'regression' 的特征列表。")
                     unit_price_pred = -1 # 使用 -1 作为错误标记
                     prediction_possible = False
                except Exception as e:
                     st.error(f"均价预测模型预测时发生意外错误: {e}")
                     unit_price_pred = -1
                     prediction_possible = False

            # --- 在主页面分列显示预测结果 ---
            if prediction_possible: # 只有在所有步骤都可能执行时才显示结果区
                st.markdown("---") # 分隔线
                st.subheader("📈 预测结果分析") # 结果区域的总标题
                col1, col2, col3 = st.columns(3) # 创建三列布局

                # 在第一列显示市场细分结果
                with col1:
                    st.markdown("<h5 style='text-align: center; color: #1f77b4; margin-bottom: 0px;'>市场细分</h5>", unsafe_allow_html=True) # 设置标题样式
                    st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True) # 添加分隔线增加视觉区分
                    if "错误" not in market_pred_label:
                        # 使用居中、大字体、加粗和颜色来突出显示结果
                        st.markdown(f"<p style='text-align: center; font-size: 28px; font-weight: bold; color: #1f77b4;'>{market_pred_label}</p>", unsafe_allow_html=True)
                        st.caption("判断房产在整体市场中的价格定位。") # 使用 caption 添加简短说明
                    else:
                        st.error(market_pred_label) # 如果预测出错，显示错误信息

                # 在第二列显示价格水平结果
                with col2:
                    st.markdown("<h5 style='text-align: center; color: #ff7f0e; margin-bottom: 0px;'>价格水平 (相对区域)</h5>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    if "错误" not in price_level_pred_label:
                         # 根据预测结果调整显示文本和颜色
                         if price_level_pred_code == 1: # 高于均价
                             display_text = "高于区域均价"
                             display_color = "#E74C3C" # 红色系，表示偏高
                         else: # 不高于均价 (包括等于)
                             display_text = "不高于区域均价"
                             display_color = "#2ECC71" # 绿色系，表示正常或偏低
                         st.markdown(f"<p style='text-align: center; font-size: 28px; font-weight: bold; color: {display_color};'>{display_text}</p>", unsafe_allow_html=True)
                         st.caption("判断房产单价是否高于其所在区域均值。")
                    else:
                         st.error(price_level_pred_label)

                # 在第三列显示均价预测结果
                with col3:
                    st.markdown("<h5 style='text-align: center; color: #2ca02c; margin-bottom: 0px;'>均价预测</h5>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    if unit_price_pred != -1: # 检查预测是否成功
                         # 使用 st.metric 组件显示数值结果，格式化输出
                         st.metric(label="预测单价 (元/㎡)", value=f"{unit_price_pred:,.0f}") # 显示为整数，带千位分隔符
                         st.caption("预测的每平方米大致价格。")
                    else:
                         st.error("无法完成房产均价预测。") # 显示错误信息

                st.success("✅ 分析预测完成！请查看上方结果。") # 显示一个成功的提示消息
                st.markdown("---") # 结束分隔线
                # 添加免责声明或提示信息
                st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
            else:
                st.warning("部分或全部预测因输入错误未能完成，请检查上方错误信息。")


# --- 应用启动时资源加载失败的处理 ---
elif not resources:
     # 如果 load_resources() 返回 None，说明加载失败
     st.error("❌ 应用程序初始化失败！")
     st.warning("无法加载必要的模型或数据文件。请检查以下几点：")
     st.markdown(f"""
        *   确认以下所有 `.joblib` 文件都与 `app.py` 文件在 **同一个** 目录下:
            *   `{MARKET_MODEL_PATH}`
            *   `{PRICE_LEVEL_MODEL_PATH}`
            *   `{REGRESSION_MODEL_PATH}`
            *   `{SCALER_PATH}`
            *   `{FEATURE_NAMES_PATH}`
            *   `{MAPPINGS_PATH}`
        *   检查运行 Streamlit 的终端是否有更详细的错误信息。
     """)

# --- 页脚信息 ---
st.sidebar.markdown("---") # 侧边栏分隔线
st.sidebar.caption("模型提供: LightGBM & RandomForest") # 模型信息
st.sidebar.caption("数据来源: 安居客") # 数据来源说明
# 你可以添加你的名字或团队信息
st.sidebar.caption("开发者: [凌欢]")