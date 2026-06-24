<!-- 铁路产品分BIN算法模块开发需求说明书
请开发一个完整的铁路产品平面度/直线度分BIN算法Python模块，实现从来料数据到BIN分类的全流程自动化。
一、项目背景与目标
1.1 产品型号
X9600系列：D-Z, D-Y, B-Z, B-Y四个方向
1.2 核心目标
将连续的测量点数据压缩为离散Shape标签（如"NNPP", "NPP"）
基于Shape统计聚类生成BIN分类
为后续整形工序提供精准参数索引
二、输入数据结构
2.1 原始测量数据格式
自行阅读当前目录的Data目录,找到Data目录中的各子目录的data.csv阅读,目录只需要读取9600下的数据，9601目录不需要处理。
数据有DZ, DY, BZ, BY四种类型。
三、算法实现步骤（五步法）
Step 1: 数据预处理与清洗
3.1 清洗规则
Python
复制
def preprocess_data(df):
    """
    输入：原始DataFrame
    处理：
    1. 删除ADD13 ≤ 0.1的行（OK品）
    2. 删除ADD13 > 0.8的行（异常料）
    3. 删除重复编码的行
    4. 返回清洗后的DataFrame
    """
    pass
3.2 生成中间变量（端点法拟合）
方法：对每行数据计算P1-P9（相对于D1-D9端点连线的偏差）
计算公式：
总增量 = D9 - D1
步长 = (D9 - D1) / 8
Pi[i] = Di - (D1 + i * 步长)，其中 i = 0~8
P1恒为0，P9≈0（浮点误差）
输出：新增列 P1 ~ P9
Step 2: 特征提取（按产品型号分段）
4.1 X9600 B-Y方向配置（二段式）
Python
复制
# 分段规则
SEGMENT_CONFIG = {
    "X9600_BY": {
        "seg1_points": ["D1", "D2", "D3", "D4", "D5", "D6", "D7"],  # 前段7点
        "seg2_point": "D8",  # 分界点
        "seg3_points": ["D8", "D9"],  # 后段2点
        "thresholds": [0, 0.015]  # t1, t2
    }
}
特征计算：
e1（主体特征）：在P1-P7范围内找最大绝对偏差
Python
复制
e1 = max([abs(P1), abs(P2), ..., abs(P7)])
e2（末端特征）：直接取P8的偏差值
Python
复制
e2 = P8
4.2 X9600 D-Z方向配置（四段式）
Python
复制
"X9600_DZ": {
    "segments": [
        {"points": ["D1","D2","D3","D4"], "method": "endpoint", "threshold": 0},  # 第1段：端点差值
        {"points": ["D5","D6","D7","D8"], "method": "straightness", "threshold": 0},  # 第2段：直线度拟合
        {"points": ["D9","D10","D11","D12","D13","D14","D15","D16","D17"], "method": "straightness", "threshold": 0},  # 第3段
        {"points": ["D18","D19","D20"], "method": "endpoint", "threshold": -0.05}  # 第4段
    ]
}
"straightness"方法实现  ：
Python
复制
def calculate_straightness(points):
   """
   输入：某段的实际测量值列表 [Pi, Pi+1, ..., Pj]
   输出：该段最大绝对偏差
   算法：端点法拟合直线，计算各点到直线的垂直偏差
   """
   # 以该段首尾为端点建立参考直线
   # 计算中间点相对直线的偏差
   # 返回 max(abs(偏差))
   pass
Step 3: 二值化标签生成
5.1 X9600 B-Y二值化规则
Python
复制
def generate_shape(e1, e2, thresholds):
    """
    thresholds = [t1, t2]
    """
    tag1 = 'P' if e1 >= thresholds[0] else 'N'
    tag2 = 'P' if e2 >= thresholds[1] else 'N'
    return tag1 + tag2  # 如"PP", "PN"
5.2 X9601系列特殊处理（三值标签）
Python
复制
def generate_shape_9601(e, t1, t2):
    """
    针对P5点等特殊位置
    t1, t2 = [-0.1, 0.1]
    """
    if e >= t2:
        return 'P'
    elif e <= t1:
        return 'N'
    else:
        return 'M'  # 中间态
输出：新增列 Shape（字符串，长度2-4字符）
Step 4: 统计分析聚类
6.1 Shape频次统计
Python
复制
def analyze_shape_distribution(df):
    """
    统计所有Shape的出现次数和占比
    返回：DataFrame [Shape, Count, Percentage]
    """
    shape_stats = df['Shape'].value_counts().reset_index()
    shape_stats.columns = ['Shape', 'Count']
    shape_stats['Percentage'] = shape_stats['Count'] / len(df) * 100
    return shape_stats
6.2 自动聚类（X9601模式）
Python
复制
def cluster_shapes_kmeans(df, n_clusters=6):
    """
    输入：包含P1-P9的DataFrame
    逻辑：
    1. 对P1-P9列做K-Means聚类
    2. 计算每类均值曲线
    3. 计算每个样本到类中心的距离
    4. 距离>0.05的标记为"BINX"
    """
    from sklearn.cluster import KMeans
    features = df[['P1','P2',...,'P9']].values
    kmeans = KMeans(n_clusters=n_clusters)
    df['Cluster'] = kmeans.fit_predict(features)
    # 后续处理...
    pass
6.3 BIN命名规则
Python
复制
BIN_NAMING_RULE = {
    "X9600_BY": {
        "PP": "BIN1",  # 占比88.59%
        "PN": "BIN2",  # 占比11.37%
        "NN": "BIN3",  # 占比0.05%
    }
}
输出：新增列 BIN（如"BIN1", "BIN2", "BINX"）
Step 5: 质量验证与拦截
7.1 类内变异系数（CV）计算
Python
复制
def calculate_intra_class_cv(df, shape):
    """
    计算指定Shape的类内CV值
    CV = σ/μ * 100%
    返回：每个点的CV列表
    """
    samples = df[df['Shape']==shape][['P1',...,'P9']].values
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    cv = std / mean * 100
    return cv
报警阈值：任意点CV > 15% → Shape过宽，需拆分
7.2 异常数据拦截（BINnX）
Python
复制
def detect_anomaly_samples(df, cv_threshold=15):
    """
    对主BIN类，检查每个样本与类均值的相似度
    相似度 = 1 - (abs(Pi - mean_i) / mean_i)
    若平均相似度 < 60% → 标记为"BINnX"
    """
    pass
四、完整函数接口定义
主入口函数
Python
复制
def binning_algorithm(input_file, product_model="X9600", direction="BY"):
    """
    主函数：从来料到BIN分类全流程
    
    参数：
    - input_file: str, CSV文件路径
    - product_model: str, "X9600"或"X9601"
    - direction: str, "D-Z", "D-Y", "B-Z", "B-Y"
    
    返回：
    - DataFrame: 包含原始数据 + P1-P9 + Shape + BIN
    
    处理流程：
    1. 读取CSV → DataFrame
    2. preprocess_data() 数据清洗
    3. generate_intermediate_vars() 计算P1-P9
    4. extract_features() 按model/direction分段
    5. generate_shape() 二值化标签
    6. analyze_distribution() Shape统计
    7. assign_bin() BIN命名
    8. validate_quality() CV计算与异常拦截
    9. 返回完整结果
    """
    pass
五、测试用例要求
用例1：X9600 B-Y基础流程
输入：9600BY.txt（2条样本）
期望输出：
Python
复制
assert output.loc[0, 'Shape'] == "PP"
assert output.loc[1, 'Shape'] == "PN"
assert output.loc[0, 'P4'] ≈ 0.08312115
用例2：OK品过滤
输入：包含ADD13=0.08的行
期望输出：该行被删除，不进入后续计算
用例3：异常值拦截
输入：包含ADD13=0.85的行
期望输出：该行被删除，并打印警告日志
用例4：类内CV报警
输入：50条Shape="PP"但P4值在0.05-0.15范围分散
期望输出：CV>15%，触发报警，建议拆分子类
六、性能与代码质量要求
性能：处理10,000条数据 < 5秒
精度：P1-P9计算误差 < 1e-10（浮点精度）
可扩展性：新增产品型号只需修改SEGMENT_CONFIG
日志：记录清洗数量、各Shape占比、CV报警信息
注释：每个函数包含docstring，关键步骤加中文注释
测试：pytest覆盖率 > 80%
七、参考文档与数据
PDF文档：
X9600Rail Binning Algorithm Flow v1.2.251119.pdf
 -->

再次生成DZ算法以版本命名，在具体版本后面用时间区别即可。
一：1，4两端的特征值计算不变。我将要修改第二段和第三段的特征值计算逻辑.
二：2，3两端修改逻辑是一致的。阅读Data\total.xlsx这个文件，基准值是SP1X，SP2X两个值的平均值，其结果需要除以100.在每段中，找到离基准值最远的点即可。仍需注意只处理状态为pre的数据。不需要进行数据清洗，直接进行数据处理即可。
三：生成数据之后再次绘制图像，按照之前提及的格式，也就是charts\detailed_comparison_lines这个路径下的格式生成对应图像。

写一个简单的验证算法，读取Data\total_final_processed.xlsx的数据。
仅采用pre状态和reshaping一个表格的数据。
按照目前的特征值计算方法，仅验证段1，段4两个段的分类结果。
最后和数据源的Shape字段作比较，统计结果，把不符合的结果的ID数字输出在一个文件中。
直接进行数据处理，不用数据预处理和数据清洗。


现在读取Data\total_final_processed.xlsx，按照我们的算法逻辑分别计算四个段的标签值，再分别和文件里的shape字段对比。生成四份报告对应四段的标签，显示差异。

现在项目很臃肿，我需要进行一定的清理和重构。
一：第一版的算法多余步骤太多，数据清晰和预处理只留下两步：将整体值（也就是类似于ADD13，FAI68等）<0.1的标记为分类BINOK，>0.8的标记为BIN100.此处多加了两种BIN分类。省去一些向量操作，直接进行核心处理，该模块rail_binning_algorithm.py仅用于提供最核心的部分，留存方法向外提供4段的特征值和分类标签。该模块的核心算法和之前保持一致：①第1、4段分别以P1-P4、 P20-P17的结果作为特征值(e)
          ②第2、3段分别进行端点法直线度拟合，将绝对值最大的元素作为特征值(e)
          ③设定每个分段的阈值(t)，如：0，0，0，-0.05
          ④每个分段的特征值(e)与阈值(t)相比较，如果e≥t，则记为标签符号‘P’，否则记为‘N’。
二：保留项目的扩展性，后续可能会扩展分类标签，而不是只有P和N；将诸如数据表字段，分类标签，BIN种类在不同类中声明，便于后续扩展和使用。

我现在要更改核心算法模块，在数据清晰和预处理后，添加一步：计算P1~P14最小二乘法拟合值结果值x,如果x<0.05,则前三个标签记为MMM，最后一个标签扔按照原计划执行。


##仅使用状态为pre和Reshaping表格的数据##

现在再生成一个画图工具，要求利用rail_binning_algorithm.py这个算法，生成完成的shape字段，然后和数据源Data\total_final_processed.xlsx中原生的shape字段做出来比较。仅使用状态为pre和Reshaping表格的数据进行比较。
生成的图像要求：根据20种BIN的分类，画出20张图，每张图用20个点位画出所有线的折线图用于观察分类效果。注意保持图的单位长度相同，一张图包含参考和生成，以及叠加观察差异的三张折线图图。

你生成的图，参考图和生成图还是一样的，这是错误的。下面我将给你流程：
1.根据当前算法rail_binning_algorithm.py分类，对于每个BIN，保存对应20个点位的所有数据。
2.读取数据源的，保存所有数据的20个点位的数据，按照BIN类型分组。
3.将生成的和参考的数据，根据两组数据的20个点位的数据，画出同一BIN类型下的两张折线图。举例来说，假设我们分类生成的数据里BIN1（PPPP）有10个数据，在数据源中有20个BIN1数据，那么我们生成一张图，这个图里面包含两个折线图，单位长度相同，分别用生成和数据源中的数据画出折线图。

根据现在的算法@rail_binning_algorithm.py，生成一组图，按照BIN种类分组。把每组的所有的20点位的数据，绘制为折线图。仅使用状态为pre和Reshaping表格的数据。不需要标注每条线对应的数据记录。


当前项目的主要目的是，先分析来料产品的分类，然后利用两个固定点将产品固定，最后决定一个或多个下压位置，执行下压量，给产品造成一个形变，使其尽可能达到最终要求。最终要求结果就是一个产品的20个点位数据，其最大和最小的数据之差小于等于0.1.
我们当前的分类并不精确，这里的精确不是强制要求所有曲线长的都很相似。精确的定义应该是，统一种类的产品，我们可以采用基本一致的处理方法，在压头的选位和下压量有很多相似的地方。当然如果相同分类的曲线基本一致，也有这个效果。
调用你的mcp分析图片的能力，分析Output\dz_visualization_simple这个路径下所有BIN对应的图像。深度思考目前的分类算法，给出可能的更精确的分类方法。目前我的想法是，由于第三段是所有产品变化最大，切点位分布最广的一段，我们可以进一步将第三段精细化分类，也就是段内细化，利用将第三段再分段或者添加更多曲线相关特征来实现。
1.你先按照我给出的建议思考进一步分类的方法。
2.抛弃我给你的分类建议，自己探索出可能有效的分类方法。
3.当你分析完图片后，将你的分析结果保存到一个文档中，以你更容易理解的方式保存即可，方便后续我提交给你当提示词。
4.我给你创建了一个research目录，你可以编写任意代码在这个目录下，用于验证想法。


根据当前算法的写法，写一个新的分类算法版本v3.段1，2，4逻辑不变，将段3再分为3段P9-P11,P12-P14,P15-P16,每段同样使用端点法直线度拟合，细化的三段阈值设置为0，0.04，0.然后新的三段分类规则也和之前一致，这意味着，新生成的shape有6个标签。注意，总体阈值设置为0，0，0，0.04，0，0.

写一个文件，将生成数据test_v3_results.csv的20个点位数据绘制成折线图，按照新的shape规则分组。把所有的点位数据全画出来，在图中标注shape的具体类型。不需要把每条线对应的记录标注，仅把线画出来即可。

我现在想到一个新的第三段的分类规则，写一个新的算法版本V4。
由于第三段涉及到8个点的数据，涵盖很广，那么我们可以将第三段本身分为4类：
ARC_UP（上圆弧）：中段单峰上凸
ARC_DOWN（下圆弧）：中段单峰下凹
FLAT（平缓型）：几乎无变形
WAVE（剧烈波动）：前后趋势剧变

特征1：整体趋势（必须）
# 总斜率 = (P17 - P9) / 8
trend = (df['P17'] - df['P9']) / 8
物理意义：判断向上/向下/平缓
ARC_UP：trend > 0.02
ARC_DOWN：trend < -0.02
FLAT/WAVE：-0.02 ≤ trend ≤ 0.02
特征2：段内标准差（核心）

# 计算P9-P17的离散程度
std_dev = df.loc[:, 'P9':'P17'].std(axis=1)
物理意义：区分平缓 vs 波动
FLAT：std_dev < 0.02
ARC_UP/ARC_DOWN：0.02 ≤ std_dev < 0.08
WAVE：std_dev ≥ 0.08

特征3：前后半段斜率差（WAVE专用）
# 识别剧烈转折
slope_left = (df['P13'] - df['P9']) / 4
slope_right = (df['P17'] - df['P13']) / 4
slope_diff = abs(slope_right - slope_left)
物理意义：捕捉"前半上、后半下"的剧变
WAVE：slope_diff > 0.1（且std_dev ≥ 0.08）
其他：slope_diff ≤ 0.1


写一个文件，将生成数据test_v4_results.csv的20个点位数据绘制成折线图，按照新的shape规则分组。把所有的点位数据全画出来，在图中标注shape的具体类型。不需要把每条线对应的记录标注，仅把线画出来即可。

写一个test_1209_DZ的文件，数据源是Data\X9600DZ\data.csv。4段，段1，段4使用端点差值法，分为是P4-P1和P20-P1，段2使用端点法直线度拟合，段3的特征值计算参考第三段分类流程特征变量总结.md。生成一个新的csv数据，将shape字段补上。段1，段2，段4阈值均为0，仅分为两类P/N。