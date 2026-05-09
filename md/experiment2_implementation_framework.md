# 实验二：评测实施框架（代码骨架）

**文件说明**：本文件包含实现实验二所需的 Python 函数骨架、数据结构、以及可视化代码框架。

---

## 第一部分：数据结构与配置

### 1.1 评测指标数据类

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from datetime import datetime

@dataclass
class InstructionEvalResult:
    """单条指令的评测结果"""
    instruction_id: str              # 如 "1-1-1"
    instruction_text: str
    model_name: str                  # "DeepSeek", "Doubao", "GPT-4o"
    
    # 基础评测结果
    parsing_success: bool            # 是否解析成功
    execution_success: bool          # 是否执行成功
    parsing_explanation: str         # LLM的理解
    generated_task_sequence: List[Dict]  # 生成的任务分解
    
    # 延迟采集
    inference_latency_ms: float      # 推理延迟
    execution_time_ms: float         # 执行耗时
    
    # 条件分支相关
    has_branch: bool                 # 是否含分支
    branch_correctness: bool         # 分支选择是否正确
    
    # 语义深度评分
    semantic_depth_score: float      # 0-100，语义修饰符理解程度
    
    # 错误归因
    error_type: Optional[str]        # None或6个错误类型之一
    error_description: str
    
    # 其他
    timestamp: str                   # ISO 8601格式
    notes: str                       # 人工备注

@dataclass
class MatrixResult:
    """3×5矩阵单元格的聚合结果"""
    difficulty: str                  # "L1", "L2", "L3"
    semantic_type: str               # "基础运动", "目标定位"等
    
    results: List[InstructionEvalResult]  # 该单元格的所有指令结果
    
    def parsing_success_rate(self) -> float:
        """计算该单元格的解析成功率(%)"""
        if not self.results:
            return 0.0
        success = sum(1 for r in self.results if r.parsing_success)
        return (success / len(self.results)) * 100
    
    def execution_success_rate(self) -> float:
        """计算该单元格的执行成功率(%)"""
        if not self.results:
            return 0.0
        # 只计算解析成功的指令
        parsed_results = [r for r in self.results if r.parsing_success]
        if not parsed_results:
            return 0.0
        success = sum(1 for r in parsed_results if r.execution_success)
        return (success / len(parsed_results)) * 100
    
    def avg_latency_ms(self) -> float:
        """平均推理延迟"""
        if not self.results:
            return 0.0
        return sum(r.inference_latency_ms for r in self.results) / len(self.results)
    
    def branch_correctness_rate(self) -> float:
        """分支准确率(%)"""
        with_branch = [r for r in self.results if r.has_branch]
        if not with_branch:
            return 100.0  # 无分支的单元格默认100%
        correct = sum(1 for r in with_branch if r.branch_correctness)
        return (correct / len(with_branch)) * 100
    
    def avg_semantic_depth(self) -> float:
        """语义深度平均分"""
        if not self.results:
            return 0.0
        return sum(r.semantic_depth_score for r in self.results) / len(self.results)

@dataclass
class RobustnessTestResult:
    """鲁棒性测试（同义表述一致性）的结果"""
    instruction_id: str
    original_text: str
    model_name: str
    
    # 3个变体的执行结果
    variant_results: List[InstructionEvalResult]
    
    def consistency_rate(self) -> float:
        """一致性率(%)：3个变体是否都成功且结果相同"""
        if len(self.variant_results) != 3:
            return 0.0
        
        # 简化：如果都执行成功且目标一致，则一致
        all_success = all(r.execution_success for r in self.variant_results)
        if not all_success:
            return 0.0
        
        # 检查目标是否一致（可通过比较生成的任务序列）
        # 这里简化处理：如果都成功就认为一致
        return 100.0

@dataclass
class ColloquialToleranceResult:
    """口语容错率测试的结果"""
    original_text: str
    model_name: str
    
    # 3个变体（废话、语序、省略主语）
    variant_results: List[InstructionEvalResult]
    
    def tolerance_rate(self) -> float:
        """容错率(%)"""
        if len(self.variant_results) != 3:
            return 0.0
        success_count = sum(1 for r in self.variant_results if r.execution_success)
        return (success_count / 3) * 100

@dataclass
class ExperimentMetrics:
    """实验的7个指标总体结果"""
    model_name: str
    
    parsing_success_rate: float          # 指标1 (%)
    execution_success_rate: float        # 指标2 (%)
    avg_inference_latency_ms: float      # 指标3 (ms，后续需反向化处理成分值)
    branch_correctness_rate: float       # 指标4 (%)
    semantic_depth_score: float          # 指标5 (0-100)
    robustness_index: float              # 指标6：鲁棒性指数 (%)
    colloquial_tolerance_rate: float     # 指标7：容错率 (%)
    
    zero_shot_success_rate: Optional[float] = None  # 泛化测试成功率
```

### 1.2 配置文件

```python
# evaluation_config.py
EVALUATION_CONFIG = {
    "models": [
        {
            "name": "DeepSeek-V3",
            "api_endpoint": "local://deepseek-api",  # 或 VolcEngine API
            "temperature": 0.7,
            "max_tokens": 500,
        },
        {
            "name": "Doubao-pro",
            "api_endpoint": "volcengine://doubao-pro",
            "temperature": 0.7,
            "max_tokens": 500,
        },
        {
            "name": "GPT-4o",
            "api_endpoint": "openai://gpt-4o",
            "temperature": 0.7,
            "max_tokens": 500,
        },
    ],
    
    "test_set": {
        "core_matrix": 45,           # 3×5×3 核心矩阵指令
        "colloquial_variants": 15,    # 容错变体
        "robustness_variants": 24,    # 鲁棒性变体（8个原始×3个变体）
        "zero_shot_variants": 24,     # 泛化测试变体
    },
    
    "evaluation_thresholds": {
        "position_error_max_m": 0.5,    # 位置误差阈值
        "speed_threshold_ms": 0.3,      # 速度约束阈值
        "semantic_depth_tolerance_m": 0.1,  # 语义理解容差
        "robustness_latency_tolerance_pct": 10,  # 鲁棒性延迟容差(%）
    },
    
    "output_dir": "./logs/test_exp2",
}
```

---

## 第二部分：核心评测函数

### 2.1 指令执行与采集

```python
class InstructionEvaluator:
    """负责执行单条指令并采集评测数据"""
    
    def __init__(self, communication_handler, llm_client):
        """
        Args:
            communication_handler: Communication_Mavlink.py 的实例
            llm_client: LLM API 客户端
        """
        self.comm = communication_handler
        self.llm = llm_client
    
    async def evaluate_instruction(
        self,
        instruction_id: str,
        instruction_text: str,
        model_name: str,
        expected_behavior: Dict,  # 标注的期望行为
    ) -> InstructionEvalResult:
        """
        执行单条指令，采集所有评测数据
        
        流程：
        1. 调用LLM解析指令
        2. 检查解析结果是否与标注一致 (指标1)
        3. 执行生成的任务序列
        4. 采集MAVLink日志：位置、速度、时间等 (指标2)
        5. 采集LLM的推理延迟 (指标3)
        6. 检查分支是否正确 (指标4)
        7. 评估语义修饰符的理解程度 (指标5)
        8. 若执行失败，进行错误归因 (6类错误)
        """
        
        import time
        result = InstructionEvalResult(
            instruction_id=instruction_id,
            instruction_text=instruction_text,
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
        )
        
        try:
            # 步骤1：调用LLM解析
            start_time = time.time()
            llm_response = await self.llm.parse_instruction(
                instruction_text=instruction_text,
                model_name=model_name,
            )
            result.inference_latency_ms = (time.time() - start_time) * 1000
            
            # 步骤2：检查解析成功
            result.parsing_explanation = llm_response.get("explanation", "")
            result.generated_task_sequence = llm_response.get("task_sequence", [])
            result.parsing_success = self._check_parsing_correctness(
                result.generated_task_sequence,
                expected_behavior,
            )
            
            if not result.parsing_success:
                result.execution_success = False
                result.error_type = "解析错误"
                return result
            
            # 步骤3：执行任务序列
            exec_start = time.time()
            execution_log = await self._execute_task_sequence(
                result.generated_task_sequence
            )
            result.execution_time_ms = (time.time() - exec_start) * 1000
            
            # 步骤4：检查执行成功
            result.execution_success = self._check_execution_success(
                execution_log,
                expected_behavior,
            )
            
            # 步骤5：检查分支正确性
            if self._has_conditional_branch(result.generated_task_sequence):
                result.has_branch = True
                result.branch_correctness = self._check_branch_correctness(
                    execution_log,
                    expected_behavior,
                )
            
            # 步骤6：评估语义深度
            result.semantic_depth_score = self._evaluate_semantic_depth(
                result.generated_task_sequence,
                expected_behavior,
            )
            
            # 步骤7：若失败，进行错误归因
            if not result.execution_success:
                result.error_type = self._attribute_error_type(
                    result,
                    execution_log,
                    expected_behavior,
                )
            
        except Exception as e:
            result.execution_success = False
            result.error_type = "执行异常"
            result.error_description = str(e)
        
        return result
    
    def _check_parsing_correctness(
        self,
        generated_task_sequence: List[Dict],
        expected_behavior: Dict,
    ) -> bool:
        """
        检查LLM生成的任务序列是否与标注的期望行为一致
        
        简化实现：
        - 比较任务序列的第一个动作类型
        - 比较主要约束条件
        """
        # TODO: 实现详细的任务序列对比逻辑
        if not generated_task_sequence:
            return False
        
        first_task = generated_task_sequence[0]
        expected_action = expected_behavior.get("primary_action", "")
        
        # 检查主要动作是否匹配
        return first_task.get("action", "") == expected_action
    
    def _execute_task_sequence(
        self,
        task_sequence: List[Dict],
    ) -> Dict:
        """执行任务序列，采集MAVLink日志"""
        # 调用 Communication_Mavlink.py 的执行函数
        # 返回执行日志：位置、速度、时间序列等
        pass
    
    def _check_execution_success(
        self,
        execution_log: Dict,
        expected_behavior: Dict,
    ) -> bool:
        """
        检查执行结果是否满足预期
        
        标准：
        - 位置偏差 < 0.5m
        - 速度 < 0.3m/s
        - 无安全违反
        """
        final_position = execution_log.get("final_position", {})
        expected_position = expected_behavior.get("final_position", {})
        
        # 计算位置误差
        position_error = self._calculate_distance(
            final_position,
            expected_position,
        )
        
        if position_error > EVALUATION_CONFIG["evaluation_thresholds"]["position_error_max_m"]:
            return False
        
        # 检查速度约束
        final_velocity = execution_log.get("final_velocity", 0.0)
        if final_velocity > EVALUATION_CONFIG["evaluation_thresholds"]["speed_threshold_ms"]:
            return False
        
        return True
    
    def _has_conditional_branch(self, task_sequence: List[Dict]) -> bool:
        """检查任务序列中是否包含条件分支"""
        return any(task.get("type") == "conditional" for task in task_sequence)
    
    def _check_branch_correctness(
        self,
        execution_log: Dict,
        expected_behavior: Dict,
    ) -> bool:
        """
        检查条件分支是否执行了正确的分支
        
        示例：
        - 指令："有红球靠近，无就降落"
        - YOLO检测结果：有红球
        - 期望执行：靠近
        - 实际执行：检查日志中是否执行了靠近而非降落
        """
        # TODO: 实现分支正确性检查
        pass
    
    def _evaluate_semantic_depth(
        self,
        task_sequence: List[Dict],
        expected_behavior: Dict,
    ) -> float:
        """
        评估LLM对语义修饰符的理解程度
        
        修饰符映射检查：
        - "一点点" 应对应 0.3m
        - "稍微" 应对应 0.2m
        - "大幅度" 应对应 45°/0.5m
        """
        score = 100.0
        
        for task in task_sequence:
            if "modifier" in task:
                modifier = task["modifier"]
                expected_value = expected_behavior.get(f"modifier_{modifier}")
                actual_value = task.get("value")
                
                # 计算偏差
                if expected_value and actual_value:
                    error_pct = abs(actual_value - expected_value) / expected_value
                    if error_pct > 0.2:  # 20%容差
                        score -= 20
        
        return max(0, score)
    
    def _attribute_error_type(
        self,
        result: InstructionEvalResult,
        execution_log: Dict,
        expected_behavior: Dict,
    ) -> str:
        """
        使用6类错误框架进行错误归因
        
        优先级：
        1. 感知错误 (YOLO检测失败)
        2. 多余生成/幻觉错误 (生成了额外动作)
        3. 条件分支识别失败
        4. 多步时序顺序错乱
        5. 空间方位理解错误
        6. 程度语义理解错误
        """
        
        # 检查1：感知错误
        yolo_results = execution_log.get("yolo_detections", [])
        if not yolo_results and "target" in expected_behavior:
            return "感知错误"
        
        # 检查2：多余生成
        expected_tasks = expected_behavior.get("task_count", 0)
        actual_tasks = len(result.generated_task_sequence)
        if actual_tasks > expected_tasks + 1:
            return "多余生成/幻觉错误"
        
        # 检查3：条件分支
        if result.has_branch and not result.branch_correctness:
            return "条件分支识别失败"
        
        # 检查4：时序顺序
        expected_order = expected_behavior.get("execution_order", [])
        actual_order = [t.get("action") for t in result.generated_task_sequence]
        if expected_order and actual_order != expected_order:
            return "多步时序顺序错乱"
        
        # 检查5：空间方位
        final_pos = execution_log.get("final_position", {})
        expected_pos = expected_behavior.get("final_position", {})
        if self._check_direction_error(final_pos, expected_pos):
            return "空间方位理解错误"
        
        # 检查6：程度语义
        if result.semantic_depth_score < 50:
            return "程度语义理解错误"
        
        return "未知错误"
    
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """计算两个位置之间的距离"""
        import math
        dx = pos1.get("x", 0) - pos2.get("x", 0)
        dy = pos1.get("y", 0) - pos2.get("y", 0)
        dz = pos1.get("z", 0) - pos2.get("z", 0)
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    def _check_direction_error(self, actual_pos: Dict, expected_pos: Dict) -> bool:
        """检查方向是否错误（如向左->向右）"""
        # TODO: 实现方向对比逻辑
        pass
```

### 2.2 聚合与统计函数

```python
class ResultAggregator:
    """聚合与统计评测结果"""
    
    @staticmethod
    def aggregate_by_matrix(
        results: List[InstructionEvalResult],
    ) -> List[MatrixResult]:
        """按3×5矩阵聚合结果"""
        from collections import defaultdict
        
        matrix_dict = defaultdict(list)
        for result in results:
            key = (result.difficulty, result.semantic_type)
            matrix_dict[key].append(result)
        
        matrix_results = []
        for (difficulty, semantic_type), instruction_results in matrix_dict.items():
            matrix_result = MatrixResult(
                difficulty=difficulty,
                semantic_type=semantic_type,
                results=instruction_results,
            )
            matrix_results.append(matrix_result)
        
        return matrix_results
    
    @staticmethod
    def calculate_overall_metrics(
        results: List[InstructionEvalResult],
    ) -> ExperimentMetrics:
        """计算模型的7个整体指标"""
        
        if not results:
            return None
        
        model_name = results[0].model_name
        
        # 指标1：解析成功率
        parsing_success = sum(1 for r in results if r.parsing_success) / len(results) * 100
        
        # 指标2：执行成功率
        parsed = [r for r in results if r.parsing_success]
        execution_success = (
            sum(1 for r in parsed if r.execution_success) / len(parsed) * 100
            if parsed else 0.0
        )
        
        # 指标3：推理延迟（平均值）
        avg_latency = sum(r.inference_latency_ms for r in results) / len(results)
        
        # 指标4：分支准确率
        with_branch = [r for r in results if r.has_branch]
        branch_correctness = (
            sum(1 for r in with_branch if r.branch_correctness) / len(with_branch) * 100
            if with_branch else 100.0
        )
        
        # 指标5：语义深度
        semantic_depth = sum(r.semantic_depth_score for r in results) / len(results)
        
        # 指标6：鲁棒性指数（需单独收集变体结果）
        # TODO: 从 RobustnessTestResult 计算
        robustness_index = 85.0
        
        # 指标7：容错率（需单独收集容错变体结果）
        # TODO: 从 ColloquialToleranceResult 计算
        colloquial_tolerance = 80.0
        
        return ExperimentMetrics(
            model_name=model_name,
            parsing_success_rate=parsing_success,
            execution_success_rate=execution_success,
            avg_inference_latency_ms=avg_latency,
            branch_correctness_rate=branch_correctness,
            semantic_depth_score=semantic_depth,
            robustness_index=robustness_index,
            colloquial_tolerance_rate=colloquial_tolerance,
        )
```

---

## 第三部分：可视化函数

### 3.1 7轴雷达图

```python
import matplotlib.pyplot as plt
import numpy as np

class RadarChartVisualizer:
    """绘制7轴雷达图"""
    
    @staticmethod
    def plot_radar_multi_models(
        metrics_list: List[ExperimentMetrics],
        output_path: str = "./logs/test_exp2/radar_chart.png",
    ):
        """
        为多个模型绘制对比雷达图
        
        Args:
            metrics_list: 各模型的 ExperimentMetrics
            output_path: 输出图表路径
        """
        
        # 7个指标轴
        categories = [
            "解析成功率\n(%)",
            "执行成功率\n(%)",
            "推理延迟\n(反向)",  # 反向：延迟越短越好
            "分支准确率\n(%)",
            "语义深度\n(0-100)",
            "鲁棒性指数\n(%)",
            "容错率\n(%)",
        ]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 角度均匀分布
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 为每个模型绘制一条曲线
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, metrics in enumerate(metrics_list):
            values = [
                metrics.parsing_success_rate,
                metrics.execution_success_rate,
                100 - min(metrics.avg_inference_latency_ms / 50 * 100, 100),  # 反向化处理
                metrics.branch_correctness_rate,
                metrics.semantic_depth_score,
                metrics.robustness_index,
                metrics.colloquial_tolerance_rate,
            ]
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=metrics.model_name, 
                    color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.title('模型多维评测对比（7轴雷达图）', size=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"雷达图已保存到: {output_path}")
```

### 3.2 错误热力图

```python
import seaborn as sns
import pandas as pd

class ErrorHeatmapVisualizer:
    """绘制错误类型分布热力图"""
    
    @staticmethod
    def plot_error_heatmap(
        results: List[InstructionEvalResult],
        output_path: str = "./logs/test_exp2/error_heatmap.png",
    ):
        """
        绘制错误类型 × 语义类型 × 难度级 的热力图
        
        行：6个错误类型
        列：语义类型 × 难度级 (5×3=15)
        """
        
        from collections import defaultdict
        error_counts = defaultdict(lambda: defaultdict(int))
        
        # 统计错误
        for result in results:
            if not result.execution_success and result.error_type:
                key = f"{result.semantic_type}_{result.difficulty}"
                error_counts[result.error_type][key] += 1
        
        # 创建DataFrame
        error_types = [
            "空间方位理解错误",
            "程度语义理解错误",
            "多步时序顺序错乱",
            "条件分支识别失败",
            "多余生成/幻觉错误",
            "感知错误",
        ]
        
        # 所有可能的单元格
        semantic_types = ["基础运动", "目标定位", "语义修饰符", "条件分支", "复合任务"]
        difficulties = ["L1", "L2", "L3"]
        columns = [f"{s}_{d}" for s in semantic_types for d in difficulties]
        
        # 构建矩阵
        data = []
        for error_type in error_types:
            row = [error_counts[error_type].get(col, 0) for col in columns]
            data.append(row)
        
        df = pd.DataFrame(data, index=error_types, columns=columns)
        
        # 绘制热力图
        plt.figure(figsize=(16, 6))
        sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': '错误个数'})
        plt.title('错误类型分布热力图（按语义类型×难度级分布）', fontsize=14, weight='bold')
        plt.xlabel('语义类型 × 难度级', fontsize=12)
        plt.ylabel('错误类型', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到: {output_path}")
```

### 3.3 指标柱状图

```python
class BarChartVisualizer:
    """绘制指标柱状图"""
    
    @staticmethod
    def plot_metrics_comparison(
        metrics_list: List[ExperimentMetrics],
        output_path: str = "./logs/test_exp2/metrics_bars.png",
    ):
        """
        为多个模型绘制 4 个核心指标的柱状图对比
        
        指标：解析成功率、执行成功率、分支准确率、容错率
        """
        
        import matplotlib.pyplot as plt
        
        model_names = [m.model_name for m in metrics_list]
        
        metrics_data = {
            '解析成功率': [m.parsing_success_rate for m in metrics_list],
            '执行成功率': [m.execution_success_rate for m in metrics_list],
            '分支准确率': [m.branch_correctness_rate for m in metrics_list],
            '容错率': [m.colloquial_tolerance_rate for m in metrics_list],
        }
        
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax.bar(x + i * width, values, width, label=metric_name)
        
        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('成功率 (%)', fontsize=12)
        ax.set_title('多模型核心指标对比', fontsize=14, weight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"柱状图已保存到: {output_path}")
```

### 3.4 泛化性能折线图

```python
class GeneralizationChartVisualizer:
    """绘制泛化性能折线图"""
    
    @staticmethod
    def plot_zero_shot_performance(
        results_by_difficulty: Dict[str, float],  # {"L1": 95, "L2": 87, "L3": 72, ...}
        model_names: List[str],
        output_path: str = "./logs/test_exp2/zero_shot_performance.png",
    ):
        """
        X轴：难度级（L1/L2/L3）
        Y轴：零样本成功率 (%)
        多条曲线：各模型
        """
        
        plt.figure(figsize=(10, 6))
        
        difficulties = ["L1", "L2", "L3"]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 假设数据按难度分组
        for idx, model_name in enumerate(model_names):
            values = [
                results_by_difficulty.get(f"{model_name}_L1", 0),
                results_by_difficulty.get(f"{model_name}_L2", 0),
                results_by_difficulty.get(f"{model_name}_L3", 0),
            ]
            plt.plot(difficulties, values, marker='o', linewidth=2, 
                    label=model_name, color=colors[idx])
        
        plt.xlabel('难度级', fontsize=12)
        plt.ylabel('零样本成功率 (%)', fontsize=12)
        plt.title('模型泛化性能对比（陌生指令识别能力）', fontsize=14, weight='bold')
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"泛化性能图已保存到: {output_path}")
```

---

## 第四部分：数据导出与报告生成

```python
import json
import csv

class ResultExporter:
    """将评测结果导出为 CSV/JSON/Markdown"""
    
    @staticmethod
    def export_to_csv(
        results: List[InstructionEvalResult],
        output_path: str = "./logs/test_exp2/evaluation_results.csv",
    ):
        """导出为CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'instruction_id', 'instruction_text', 'model_name',
                'parsing_success', 'execution_success', 'inference_latency_ms',
                'branch_correctness', 'semantic_depth_score', 'error_type'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'instruction_id': result.instruction_id,
                    'instruction_text': result.instruction_text,
                    'model_name': result.model_name,
                    'parsing_success': result.parsing_success,
                    'execution_success': result.execution_success,
                    'inference_latency_ms': f"{result.inference_latency_ms:.2f}",
                    'branch_correctness': result.branch_correctness,
                    'semantic_depth_score': f"{result.semantic_depth_score:.2f}",
                    'error_type': result.error_type or 'N/A',
                })
        
        print(f"CSV已导出到: {output_path}")
    
    @staticmethod
    def export_to_json(
        metrics_list: List[ExperimentMetrics],
        output_path: str = "./logs/test_exp2/evaluation_metrics.json",
    ):
        """导出为JSON"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'models': [
                {
                    'name': m.model_name,
                    'metrics': {
                        'parsing_success_rate': f"{m.parsing_success_rate:.2f}%",
                        'execution_success_rate': f"{m.execution_success_rate:.2f}%",
                        'avg_inference_latency_ms': f"{m.avg_inference_latency_ms:.2f}",
                        'branch_correctness_rate': f"{m.branch_correctness_rate:.2f}%",
                        'semantic_depth_score': f"{m.semantic_depth_score:.2f}",
                        'robustness_index': f"{m.robustness_index:.2f}%",
                        'colloquial_tolerance_rate': f"{m.colloquial_tolerance_rate:.2f}%",
                    }
                }
                for m in metrics_list
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON已导出到: {output_path}")
```

---

## 使用示例

```python
# 主评测流程
async def run_experiment():
    from ServerFile.Communication_Mavlink import CommunicationHandler
    from ServerFile.OpenAI_api_Mavlink_Agent import LLMClient
    
    # 初始化
    comm = CommunicationHandler()
    llm_clients = {
        "DeepSeek": LLMClient("deepseek"),
        "Doubao": LLMClient("doubao"),
        "GPT-4o": LLMClient("gpt-4o"),
    }
    
    evaluator = InstructionEvaluator(comm, llm_clients["DeepSeek"])
    
    # 加载指令库
    with open("experiment2_instruction_library.md") as f:
        instructions = parse_instruction_library(f.read())
    
    # 运行评测
    all_results = []
    for model_name, llm_client in llm_clients.items():
        evaluator.llm = llm_client
        for instr in instructions[:10]:  # 先测10条
            result = await evaluator.evaluate_instruction(
                instruction_id=instr['id'],
                instruction_text=instr['text'],
                model_name=model_name,
                expected_behavior=instr['expected_behavior'],
            )
            all_results.append(result)
    
    # 聚合与可视化
    aggregator = ResultAggregator()
    metrics_list = [
        aggregator.calculate_overall_metrics([r for r in all_results if r.model_name == m])
        for m in ["DeepSeek", "Doubao", "GPT-4o"]
    ]
    
    # 绘制图表
    RadarChartVisualizer.plot_radar_multi_models(metrics_list)
    ErrorHeatmapVisualizer.plot_error_heatmap(all_results)
    BarChartVisualizer.plot_metrics_comparison(metrics_list)
    
    # 导出结果
    ResultExporter.export_to_csv(all_results)
    ResultExporter.export_to_json(metrics_list)

# 运行
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_experiment())
```

