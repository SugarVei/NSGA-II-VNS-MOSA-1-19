# -*- coding: utf-8 -*-
"""
算法对比试验窗口
Algorithm Comparison Window

提供算法选择、多算例配置、运行对比试验和结果展示功能。
支持多个问题规模同时对比。
"""

import sys
import os
import time
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QProgressBar, QGroupBox, QScrollArea, QWidget,
    QHeaderView, QMessageBox, QFileDialog, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.comparison_worker import ComparisonWorker
from ui.case_data import CaseConfig, get_default_cases, get_default_algorithm_params


# 样式定义 - 行高翻倍版本
COMPARISON_STYLE = """
QDialog {
    background-color: #F1F8E9;
    font-family: "Microsoft YaHei", sans-serif;
}
QGroupBox {
    font-size: 16px;
    font-weight: bold;
    color: #1B5E20;
    border: 2px solid #A5D6A7;
    border-radius: 8px;
    margin-top: 20px;
    padding-top: 30px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 10px;
}
QLabel {
    color: #1B5E20;
    font-size: 14px;
    min-height: 28px;
    padding: 4px 0;
}
QCheckBox {
    color: #1B5E20;
    font-size: 14px;
    spacing: 16px;
    min-height: 40px;
    padding: 8px 0;
}
QCheckBox::indicator {
    width: 24px;
    height: 24px;
}
QPushButton {
    background-color: #43A047;
    color: white;
    border: none;
    padding: 20px 40px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: bold;
    min-height: 40px;
}
QPushButton:hover {
    background-color: #388E3C;
}
QPushButton:disabled {
    background-color: #A5D6A7;
}
QPushButton#selectCasesBtn {
    background-color: #1976D2;
    font-size: 15px;
    padding: 24px 48px;
}
QPushButton#selectCasesBtn:hover {
    background-color: #1565C0;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    padding: 16px;
    border: 2px solid #A5D6A7;
    border-radius: 6px;
    background-color: white;
    font-size: 14px;
    min-height: 36px;
}
QTableWidget {
    background-color: white;
    border: 2px solid #A5D6A7;
    border-radius: 6px;
    gridline-color: #E8F5E9;
}
QTableWidget::item {
    padding: 16px;
}
QHeaderView::section {
    background-color: #43A047;
    color: white;
    padding: 16px;
    border: none;
    font-weight: bold;
    min-height: 32px;
}
QTextEdit {
    background-color: #263238;
    color: #B0BEC5;
    border: 2px solid #A5D6A7;
    border-radius: 6px;
    font-family: Consolas, monospace;
    font-size: 12px;
    line-height: 24px;
}
QProgressBar {
    border: 2px solid #A5D6A7;
    border-radius: 6px;
    text-align: center;
    background-color: #E8F5E9;
    min-height: 32px;
}
QProgressBar::chunk {
    background-color: #43A047;
    border-radius: 4px;
}
QTabWidget::pane {
    border: 2px solid #A5D6A7;
    border-radius: 6px;
    background-color: white;
}
QTabBar::tab {
    background-color: #C8E6C9;
    color: #1B5E20;
    padding: 20px 40px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background-color: #43A047;
    color: white;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    border: none;
    background-color: #E8F5E9;
    width: 14px;
    margin: 0px;
    border-radius: 7px;
}
QScrollBar::handle:vertical {
    background-color: #81C784;
    min-height: 30px;
    border-radius: 7px;
}
QScrollBar::handle:vertical:hover {
    background-color: #66BB6A;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    border: none;
    background-color: #E8F5E9;
    height: 14px;
    margin: 0px;
    border-radius: 7px;
}
QScrollBar::handle:horizontal {
    background-color: #81C784;
    min-width: 30px;
    border-radius: 7px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #66BB6A;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""


class ParameterDialog(QDialog):
    """算法参数编辑对话框"""
    
    def __init__(self, parent, alg_name: str, current_params: dict):
        super().__init__(parent)
        self.alg_name = alg_name
        self.params = current_params.copy()
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle(f"{self.alg_name} 参数设置")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # 参数输入区
        grid = QGridLayout()
        self.inputs = {}
        
        row = 0
        for param_name, value in self.params.items():
            label = QLabel(param_name)
            grid.addWidget(label, row, 0)
            
            if isinstance(value, float):
                spinner = QDoubleSpinBox()
                spinner.setRange(0.0, 10000.0)
                spinner.setDecimals(3)
                spinner.setValue(value)
                self.inputs[param_name] = spinner
            else:
                spinner = QSpinBox()
                spinner.setRange(1, 10000)
                spinner.setValue(int(value))
                self.inputs[param_name] = spinner
            
            grid.addWidget(spinner, row, 1)
            row += 1
        
        layout.addLayout(grid)
        
        # 按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
    
    def get_params(self) -> dict:
        """获取修改后的参数"""
        result = {}
        for param_name, spinner in self.inputs.items():
            result[param_name] = spinner.value()
        return result


class AlgorithmComparisonWindow(QDialog):
    """
    算法对比试验窗口
    
    功能：
    - 选择多个问题规模（算例）
    - 选择要对比的算法
    - 配置每个算法的参数
    - 运行对比试验
    - 展示 IGD/HV/GD 的 Mean±Std 结果（按算例分列）
    """
    
    # 固定8个算法
    ALGORITHMS = [
        'MOEA/D', 'SPEA2', 'MOPSO', 'NSGA-II',
        'NSGA2-VNS', 'NSGA2-MOSA', 'MOSA', 'NSGA2-VNS-MOSA'
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.params_dict = get_default_algorithm_params()
        self.cases = []  # 已选择的算例列表
        self.worker = None
        self.results = None  # results[case_no][alg_name] = {...}
        self.start_time = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer_display)
        self.last_info = None  # 存储最近收到的进度信息
        self.setup_ui()
        self.setStyleSheet(COMPARISON_STYLE)
        
        # 设置窗口标志：添加最大化和最小化按钮
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

    
    def update_timer_display(self):
        """计时器槽函数：实时更新已用时间"""
        if not self.start_time:
            return
        
        elapsed = time.time() - self.start_time
        from ui.comparison_worker import format_time
        elapsed_str = format_time(elapsed)
        
        # 更新时间显示，保留预估剩余部分
        rem_str = "计算中..."
        if self.last_info and 'remaining_str' in self.last_info:
            rem_str = self.last_info['remaining_str']
            
        self.time_info_label.setText(
            f"⬜ 已用: {elapsed_str} | "
            f"⬜ 预估剩余: {rem_str}"
        )
    
    def setup_ui(self):
        self.setWindowTitle("算法对比试验")
        self.setMinimumSize(1200, 900)
        
        # 创建主滚动区域
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建滚动内容容器
        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setSpacing(24)  # 增加组件之间的间距
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # ===== 顶部：多问题规模选择 =====
        case_group = QGroupBox("问题规模配置")
        case_layout = QHBoxLayout(case_group)
        
        self.select_cases_btn = QPushButton("📋 选择多个问题规模")
        self.select_cases_btn.setObjectName("selectCasesBtn")
        self.select_cases_btn.clicked.connect(self.open_case_manager)
        case_layout.addWidget(self.select_cases_btn)
        
        self.cases_info_label = QLabel("尚未选择任何算例")
        self.cases_info_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #F57C00;"
        )
        case_layout.addWidget(self.cases_info_label)
        
        case_layout.addStretch()
        main_layout.addWidget(case_group)
        
        # ===== 中部：算法选择和参数 =====
        alg_group = QGroupBox("算法选择（8个对比算法）")
        alg_layout = QVBoxLayout(alg_group)
        
        self.alg_checkboxes = {}
        self.param_buttons = {}
        
        grid = QGridLayout()
        for i, alg in enumerate(self.ALGORITHMS):
            row = i // 2
            col = (i % 2) * 3
            
            cb = QCheckBox(alg)
            cb.setChecked(True)  # 默认全选
            self.alg_checkboxes[alg] = cb
            grid.addWidget(cb, row, col)
            
            param_btn = QPushButton("参数")
            param_btn.setFixedWidth(120)  # 从 60 调整为 120
            param_btn.clicked.connect(lambda checked, a=alg: self.edit_params(a))
            self.param_buttons[alg] = param_btn
            grid.addWidget(param_btn, row, col + 1)
        
        # 为列设置间距，确保不会重叠
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 2)
        grid.setColumnStretch(4, 1)
        
        alg_layout.addLayout(grid)
        
        # 全选/全不选按钮
        select_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.setFixedWidth(120)  # 从 60 调整为 120
        select_all_btn.clicked.connect(lambda: self._set_all_algorithms(True))
        select_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("取消全选")
        select_none_btn.setFixedWidth(160)  # 从 80 调整为 160
        select_none_btn.clicked.connect(lambda: self._set_all_algorithms(False))
        select_layout.addWidget(select_none_btn)
        select_layout.addStretch()
        alg_layout.addLayout(select_layout)
        
        main_layout.addWidget(alg_group)
        
        # ===== 运行参数 =====
        run_group = QGroupBox("运行参数")
        run_group_layout = QVBoxLayout(run_group)
        
        # 第一行：基础运行参数
        run_layout = QHBoxLayout()
        
        run_layout.addWidget(QLabel("每算例重复次数:"))
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 100)
        self.runs_spin.setValue(30)  # 默认30次
        run_layout.addWidget(self.runs_spin)
        
        run_layout.addWidget(QLabel("随机种子:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        run_layout.addWidget(self.seed_spin)
        
        run_layout.addStretch()
        
        self.start_btn = QPushButton("🚀 开始对比")
        self.start_btn.setFixedWidth(300)  # 从 150 调整为 300
        self.start_btn.clicked.connect(self.start_comparison)
        run_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setFixedWidth(160)  # 从 80 调整为 160
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_comparison)
        run_layout.addWidget(self.cancel_btn)
        
        run_group_layout.addLayout(run_layout)
        
        # 第二行：目标权重参数
        weight_layout = QHBoxLayout()
        
        weight_layout.addWidget(QLabel("目标权重（用于计算综合值）:"))
        
        weight_layout.addWidget(QLabel("F1(完工时间):"))
        self.w1_spin = QDoubleSpinBox()
        self.w1_spin.setRange(0.0, 1.0)
        self.w1_spin.setDecimals(2)
        self.w1_spin.setSingleStep(0.1)
        self.w1_spin.setValue(0.4)
        self.w1_spin.setFixedWidth(80)
        weight_layout.addWidget(self.w1_spin)
        
        weight_layout.addWidget(QLabel("F2(能耗):"))
        self.w2_spin = QDoubleSpinBox()
        self.w2_spin.setRange(0.0, 1.0)
        self.w2_spin.setDecimals(2)
        self.w2_spin.setSingleStep(0.1)
        self.w2_spin.setValue(0.3)
        self.w2_spin.setFixedWidth(80)
        weight_layout.addWidget(self.w2_spin)
        
        weight_layout.addWidget(QLabel("F3(人工成本):"))
        self.w3_spin = QDoubleSpinBox()
        self.w3_spin.setRange(0.0, 1.0)
        self.w3_spin.setDecimals(2)
        self.w3_spin.setSingleStep(0.1)
        self.w3_spin.setValue(0.3)
        self.w3_spin.setFixedWidth(80)
        weight_layout.addWidget(self.w3_spin)
        
        weight_layout.addStretch()
        run_group_layout.addLayout(weight_layout)
        
        main_layout.addWidget(run_group)

        
        # ===== 进度区 =====
        progress_group = QGroupBox("运行进度")
        progress_layout = QVBoxLayout(progress_group)
        
        # 详细进度信息行
        detail_layout = QHBoxLayout()
        
        # 左侧：当前任务信息
        self.task_info_label = QLabel("等待开始...")
        self.task_info_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1976D2;"
        )
        detail_layout.addWidget(self.task_info_label)
        
        detail_layout.addStretch()
        
        # 右侧：时间信息
        self.time_info_label = QLabel("")
        self.time_info_label.setStyleSheet(
            "font-size: 13px; color: #616161;"
        )
        detail_layout.addWidget(self.time_info_label)
        
        progress_layout.addLayout(detail_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)
        
        # 算法进度细节
        algo_detail_layout = QHBoxLayout()
        self.algo_progress_label = QLabel("")
        self.algo_progress_label.setStyleSheet("font-size: 12px; color: #757575;")
        algo_detail_layout.addWidget(self.algo_progress_label)
        algo_detail_layout.addStretch()
        self.run_progress_label = QLabel("")
        self.run_progress_label.setStyleSheet("font-size: 12px; color: #757575;")
        algo_detail_layout.addWidget(self.run_progress_label)
        progress_layout.addLayout(algo_detail_layout)
        
        main_layout.addWidget(progress_group)
        
        # ===== 日志区 =====
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)
        
        # ===== 结果展示区（Tab 形式） =====
        result_group = QGroupBox("对比结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_tabs = QTabWidget()
        
        # IGD Tab
        self.igd_table = QTableWidget()
        self.result_tabs.addTab(self.igd_table, "IGD (↓越小越好)")
        
        # HV Tab
        self.hv_table = QTableWidget()
        self.result_tabs.addTab(self.hv_table, "HV (↑越大越好)")
        
        # GD Tab
        self.gd_table = QTableWidget()
        self.result_tabs.addTab(self.gd_table, "GD (↓越小越好)")
        
        # Composite Tab (综合值)
        self.composite_table = QTableWidget()
        self.result_tabs.addTab(self.composite_table, "综合值 (↓越小越好)")

        
        result_layout.addWidget(self.result_tabs)
        
        # 自动保存路径显示
        self.auto_save_label = QLabel("")
        self.auto_save_label.setStyleSheet(
            "font-size: 13px; color: #2E7D32; font-weight: bold; padding: 8px; "
            "background-color: #E8F5E9; border-radius: 4px;"
        )
        self.auto_save_label.setWordWrap(True)
        self.auto_save_label.setVisible(False)
        result_layout.addWidget(self.auto_save_label)
        
        # 导出按钮
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton("📥 导出 CSV")
        self.export_btn.setFixedWidth(200)  # 设置固定宽度并调大
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        export_layout.addWidget(self.export_btn)
        result_layout.addLayout(export_layout)
        
        main_layout.addWidget(result_group)
        
        # 设置滚动区域
        scroll_area.setWidget(scroll_content)
        
        # 创建外层布局
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll_area)
    
    def _set_all_algorithms(self, checked: bool):
        """设置所有算法的选中状态"""
        for cb in self.alg_checkboxes.values():
            cb.setChecked(checked)
    
    def open_case_manager(self):
        """打开多算例管理对话框"""
        from ui.multi_case_manager import MultiCaseManagerDialog
        
        dialog = MultiCaseManagerDialog(self, self.cases if self.cases else None)
        
        if dialog.exec_() == QDialog.Accepted:
            self.cases = dialog.get_all_cases()
            self._update_cases_info()
    
    def _update_cases_info(self):
        """更新已选算例信息显示"""
        if not self.cases:
            self.cases_info_label.setText("尚未选择任何算例")
            self.cases_info_label.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #F57C00;"
            )
        else:
            configured = sum(1 for c in self.cases if c.is_configured)
            text = f"已选择 {len(self.cases)} 个算例"
            if configured < len(self.cases):
                text += f" (其中 {configured} 个已配置)"
            self.cases_info_label.setText(text)
            self.cases_info_label.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #2E7D32;"
            )
    
    def edit_params(self, alg_name: str):
        """打开参数编辑对话框"""
        dialog = ParameterDialog(self, alg_name, self.params_dict[alg_name])
        if dialog.exec_() == QDialog.Accepted:
            self.params_dict[alg_name] = dialog.get_params()
            self.log_text.append(f"已更新 {alg_name} 参数")
    
    def get_selected_algorithms(self) -> list:
        """获取选中的算法列表"""
        selected = []
        for alg, cb in self.alg_checkboxes.items():
            if cb.isChecked():
                selected.append(alg)
        return selected
    
    def start_comparison(self):
        """开始对比试验"""
        selected = self.get_selected_algorithms()
        if not selected:
            QMessageBox.warning(self, "警告", "请至少选择一个算法")
            return
        
        if not self.cases:
            QMessageBox.warning(self, "警告", "请先选择问题规模（算例）")
            return
        
        runs = self.runs_spin.value()
        seed = self.seed_spin.value()
        
        # 保存选中的算法顺序
        self.selected_algorithms_order = selected
        
        # 计算总任务数并显示预估信息
        total_tasks = len(self.cases) * len(selected) * runs
        self.log_text.clear()
        self.log_text.append(f"选中算法: {', '.join(selected)}")
        self.log_text.append(f"算例数量: {len(self.cases)}, 每算例重复: {runs} 次, 种子: {seed}")
        self.log_text.append(f"总运行次数: {total_tasks}")
        
        # 重置进度标签
        self.task_info_label.setText("🔧 正在准备试验环境 (检查配置 → 生成任务 → 创建进程池)...")
        self.time_info_label.setText("")
        self.algo_progress_label.setText("")
        self.run_progress_label.setText("")
        self.progress_bar.setValue(0)
        
        # 禁用开始按钮
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        
        # 创建并启动工作线程
        self.worker = ComparisonWorker(
            selected_algorithms=selected,
            cases_config=self.cases,
            params_dict=self.params_dict,
            runs=runs,
            base_seed=seed,
            weights=(self.w1_spin.value(), self.w2_spin.value(), self.w3_spin.value())
        )
        
        self.worker.progress.connect(self.on_progress)
        self.worker.detailed_progress.connect(self.on_detailed_progress)
        self.worker.log.connect(self.on_log)
        self.worker.finished_result.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        
        # 启动计时器
        self.start_time = time.time()
        self.timer.start(1000)  # 每秒更新一次
        
        self.worker.start()
    
    def cancel_comparison(self):
        """取消对比试验"""
        if self.worker:
            self.worker.cancel()
            self.log_text.append("正在取消...")
    
    def on_progress(self, current: int, total: int, message: str):
        """进度更新"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        percent = (current / total * 100) if total > 0 else 0
        self.progress_bar.setFormat(f"{percent:.1f}% ({current}/{total})")
    
    def on_detailed_progress(self, info: dict):
        """详细进度更新"""
        self.last_info = info  # 存储以供计时器使用
        
        # 更新任务信息
        self.task_info_label.setText(
            f"📊 Case {info['case_no']} ({info['case_scale']}) | "
            f"算法: {info['algorithm']} | "
            f"第 {info['run_idx']}/{info['runs_total']} 次"
        )
        
        # 立即更新一次时间信息
        self.time_info_label.setText(
            f"⬜ 已用: {info['elapsed_str']} | "
            f"⬜ 预估剩余: {info['remaining_str']}"
        )
        
        # 更新算法进度
        alg_idx = self.selected_algorithms_order.index(info['algorithm']) + 1 if hasattr(self, 'selected_algorithms_order') else 0
        self.algo_progress_label.setText(
            f"算例进度: {info.get('case_no', 0)}/{info['n_cases']} | "
            f"算法: {info['algorithm']} ({info['n_algorithms']}个对比算法)"
        )
        
        self.run_progress_label.setText(
            f"总进度: {info['current']}/{info['total']} ({info['percent']:.1f}%)"
        )
    
    def on_log(self, message: str):
        """日志更新"""
        self.log_text.append(message)
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def on_finished(self, results: dict):
        """试验完成"""
        self.results = results
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.export_btn.setEnabled(True)
        
        # 更新进度标签为完成状态
        self.task_info_label.setText("✅ 试验完成!")
        self.task_info_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2E7D32;"
        )
        self.algo_progress_label.setText("")
        self.run_progress_label.setText("")
        self.timer.stop()  # 停止计时器
        
        # 设置表格
        self._setup_result_tables()
        
        # 自动保存结果
        saved_path = self.auto_save_results()
        if saved_path:
            self.auto_save_label.setText(f"📁 已自动保存到: {saved_path}")
            self.auto_save_label.setVisible(True)
            self.log_text.append(f"\n💾 结果已自动保存到: {saved_path}")
        
        self.log_text.append("\n✅ 结果表格已更新")
        self.progress_bar.setValue(self.progress_bar.maximum())
    
    def auto_save_results(self) -> str:
        """自动保存结果到CSV文件
        
        Returns:
            str: 保存的文件路径，失败时返回空字符串
        """
        if not self.results:
            return ""
        
        try:
            # 确定保存目录：项目根目录/results/comparison_results
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(project_root, "results", "comparison_results")
            
            # 创建目录（如果不存在）
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"comparison_{timestamp}.csv"
            file_path = os.path.join(save_dir, file_name)
            
            # 写入汇总CSV文件（保持原有格式）
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Case_No,Problem_Scale,Algorithm,IGD_Mean,IGD_Std,HV_Mean,HV_Std,GD_Mean,GD_Std,Composite_Mean,Composite_Std,Valid_Runs\n")
                
                for case_no in sorted(self.results.keys()):
                    case = next((c for c in self.cases if c.case_no == case_no), None)
                    scale_str = case.problem_scale_str if case else "Unknown"
                    
                    for alg_name, metrics in self.results[case_no].items():
                        f.write(
                            f"{case_no},{scale_str},{alg_name},"
                            f"{metrics.get('igd_mean', 0):.6f},{metrics.get('igd_std', 0):.6f},"
                            f"{metrics.get('hv_mean', 0):.6f},{metrics.get('hv_std', 0):.6f},"
                            f"{metrics.get('gd_mean', 0):.6f},{metrics.get('gd_std', 0):.6f},"
                            f"{metrics.get('composite_mean', 0):.6f},{metrics.get('composite_std', 0):.6f},"
                            f"{metrics.get('n_valid_runs', 0)}\n"
                        )
            
            # 额外保存 per-run 长格式CSV（用于 Wilcoxon 秩和检验）
            per_run_file = os.path.join(save_dir, f"comparison_{timestamp}_per_run.csv")
            self._save_per_run_csv(per_run_file)
            self.log_text.append(f"💾 Per-run 数据已保存到: {per_run_file}")
            
            return file_path
            
        except Exception as e:
            self.log_text.append(f"⚠️ 自动保存失败: {str(e)}")
            return ""
    
    def _save_per_run_csv(self, file_path: str):
        """保存每次独立运行的原始指标值到长格式CSV文件
        
        输出格式：每行对应一次独立运行，直接兼容 Wilcoxon 秩和检验。
        
        Args:
            file_path: CSV文件保存路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("Algorithm,Instance,Run,IGD,HV,GD,Composite\n")
            
            for case_no in sorted(self.results.keys()):
                for alg_name, metrics in self.results[case_no].items():
                    igd_vals = metrics.get('igd_values', [])
                    hv_vals = metrics.get('hv_values', [])
                    gd_vals = metrics.get('gd_values', [])
                    comp_vals = metrics.get('composite_values', [])
                    
                    n_runs = len(igd_vals)
                    for run_idx in range(n_runs):
                        f.write(
                            f"{alg_name},{case_no},{run_idx + 1},"
                            f"{igd_vals[run_idx]:.8f},"
                            f"{hv_vals[run_idx]:.8f},"
                            f"{gd_vals[run_idx]:.8f},"
                            f"{comp_vals[run_idx]:.8f}\n"
                        )
    
    def _setup_result_tables(self):
        """设置结果表格 - 论文级展示优化"""
        if not self.results:
            return
        
        # 获取算例列表和算法列表
        case_nos = sorted(self.results.keys())
        selected_algs = self.get_selected_algorithms()
        
        for table, metric_key, is_lower_better in [
            (self.igd_table, 'igd', True),
            (self.hv_table, 'hv', False),
            (self.gd_table, 'gd', True),
            (self.composite_table, 'composite', True),
        ]:
            table.clear()
            table.setRowCount(len(selected_algs))
            # 每个 Case 占两列: Mean 和 Std
            table.setColumnCount(len(case_nos) * 2)
            
            # 设置列头
            headers = []
            for cn in case_nos:
                headers.append(f"Case {cn}\n(Mean)")
                headers.append(f"Case {cn}\n(Std)")
            table.setHorizontalHeaderLabels(headers)
            
            # 设置行头（算法名）
            table.setVerticalHeaderLabels(selected_algs)
            
            # 填充数据
            best_per_case = {}  # 每个 Case 的最佳 Mean 所在行
            
            for c_idx, case_no in enumerate(case_nos):
                best_val = float('inf') if is_lower_better else float('-inf')
                best_row = -1
                
                col_mean = c_idx * 2
                col_std = c_idx * 2 + 1
                
                for row, alg_name in enumerate(selected_algs):
                    if alg_name in self.results.get(case_no, {}):
                        metrics = self.results[case_no][alg_name]
                        mean_val = metrics.get(f'{metric_key}_mean', float('nan'))
                        std_val = metrics.get(f'{metric_key}_std', 0.0)
                        
                        # Mean 单元格 - 科学计数法
                        mean_item = QTableWidgetItem(f"{mean_val:.4e}")
                        mean_item.setTextAlignment(Qt.AlignCenter)
                        table.setItem(row, col_mean, mean_item)
                        
                        # Std 单元格 - 科学计数法
                        std_item = QTableWidgetItem(f"{abs(std_val):.4e}")
                        std_item.setTextAlignment(Qt.AlignCenter)
                        std_item.setForeground(QColor("#666666"))  # 标准差颜色稍淡
                        table.setItem(row, col_std, std_item)
                        
                        # 记录最佳 Mean 值
                        if is_lower_better:
                            if mean_val < best_val:
                                best_val = mean_val
                                best_row = row
                        else:
                            if mean_val > best_val:
                                best_val = mean_val
                                best_row = row
                    else:
                        # N/A 情况
                        for col in [col_mean, col_std]:
                            item = QTableWidgetItem("N/A")
                            item.setTextAlignment(Qt.AlignCenter)
                            item.setForeground(QColor(150, 150, 150))
                            table.setItem(row, col, item)
                
                best_per_case[c_idx] = best_row
            
            # 高亮最佳值和表头样式渲染
            for c_idx, best_row in best_per_case.items():
                if best_row >= 0:
                    # 仅高亮 Mean 单元格，并加粗字体
                    item = table.item(best_row, c_idx * 2)
                    if item:
                        item.setBackground(QColor("#C8E6C9")) # 浅绿色背景
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                        item.setForeground(QColor("#1B5E20")) # 深绿色文字
            
            # 优化表头整体高度
            table.horizontalHeader().setMinimumSectionSize(100)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    def on_error(self, error_message: str):
        """错误处理"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        # 更新进度标签为错误状态
        self.task_info_label.setText("❌ 发生错误")
        self.task_info_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #D32F2F;"
        )
        
        self.log_text.append(f"❌ 错误: {error_message}")
        self.timer.stop()  # 停止计时器
        QMessageBox.critical(self, "错误", error_message)
    
    def export_results(self):
        """导出结果到 CSV（汇总表 + per-run 长格式表）"""
        if not self.results:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "algorithm_comparison_multi.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # 1. 导出汇总表（保持原有格式）
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Case_No,Problem_Scale,Algorithm,IGD_Mean,IGD_Std,HV_Mean,HV_Std,GD_Mean,GD_Std,Composite_Mean,Composite_Std,Valid_Runs\n")
                    
                    for case_no in sorted(self.results.keys()):
                        case = next((c for c in self.cases if c.case_no == case_no), None)
                        scale_str = case.problem_scale_str if case else "Unknown"
                        
                        for alg_name, metrics in self.results[case_no].items():
                            f.write(
                                f"{case_no},{scale_str},{alg_name},"
                                f"{metrics.get('igd_mean', 0):.6f},{metrics.get('igd_std', 0):.6f},"
                                f"{metrics.get('hv_mean', 0):.6f},{metrics.get('hv_std', 0):.6f},"
                                f"{metrics.get('gd_mean', 0):.6f},{metrics.get('gd_std', 0):.6f},"
                                f"{metrics.get('composite_mean', 0):.6f},{metrics.get('composite_std', 0):.6f},"
                                f"{metrics.get('n_valid_runs', 0)}\n"
                            )
                
                # 2. 额外导出 per-run 长格式CSV（用于 Wilcoxon 检验）
                base, ext = os.path.splitext(file_path)
                per_run_path = f"{base}_per_run{ext}"
                self._save_per_run_csv(per_run_path)
                
                QMessageBox.information(
                    self, "导出成功",
                    f"汇总结果已导出到:\n{file_path}\n\n"
                    f"Per-run 原始数据（Wilcoxon检验用）已导出到:\n{per_run_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
