"""
结果展示面板模块
Result Panel Module

展示优化结果、图表和导出功能，采用现代化设计风格。
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QLabel, QPushButton, QTextEdit,
    QFrame, QSplitter, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

from typing import List, Optional, Dict
import numpy as np
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution
from visualization.convergence import plot_convergence, plot_comparison
from visualization.pareto_3d import plot_pareto_3d, plot_pareto_2d_projections
from visualization.export import export_pareto_to_csv, generate_report


class ResultPanel(QWidget):
    """
    结果展示面板
    
    包含图表展示、数值结果和导出功能。
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.pareto_solutions: List[Solution] = []
        self.convergence_data: Dict = {}
        self.current_figures: Dict[str, Figure] = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #E1E4E8;
                border-radius: 8px;
                background-color: #FFFFFF;
            }
            QTabBar::tab {
                background-color: #F6F8FA;
                border: 1px solid #E1E4E8;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 20px;
                margin-right: 4px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                border-bottom: 2px solid #1565C0;
                color: #1565C0;
            }
            QTabBar::tab:hover:!selected {
                background-color: #E3F2FD;
            }
        """)
        
        # Tab 1: Pareto前沿图
        self.pareto_tab = self._create_pareto_tab()
        self.tab_widget.addTab(self.pareto_tab, "Pareto前沿")
        
        # Tab 2: 收敛曲线
        self.convergence_tab = self._create_convergence_tab()
        self.tab_widget.addTab(self.convergence_tab, "收敛曲线")
        
        # Tab 3: 数值结果
        self.results_tab = self._create_results_tab()
        self.tab_widget.addTab(self.results_tab, "数值结果")
        
        # Tab 4: 日志
        self.log_tab = self._create_log_tab()
        self.tab_widget.addTab(self.log_tab, "运行日志")
        
        layout.addWidget(self.tab_widget)
        
        # 底部导出按钮区域
        export_frame = QFrame()
        export_frame.setStyleSheet("""
            QFrame {
                background-color: #F6F8FA;
                border: 1px solid #E1E4E8;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        export_layout = QHBoxLayout(export_frame)
        export_layout.setContentsMargins(12, 8, 12, 8)
        export_layout.setSpacing(12)
        
        export_label = QLabel("导出选项:")
        export_label.setStyleSheet("color: #5C6370; font-weight: 500;")
        export_layout.addWidget(export_label)
        
        self.export_csv_btn = QPushButton("导出CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.setStyleSheet(self._get_export_button_style())
        
        self.export_plots_btn = QPushButton("保存图表")
        self.export_plots_btn.clicked.connect(self.export_plots)
        self.export_plots_btn.setEnabled(False)
        self.export_plots_btn.setStyleSheet(self._get_export_button_style())
        
        self.export_report_btn = QPushButton("生成报告")
        self.export_report_btn.clicked.connect(self.export_report)
        self.export_report_btn.setEnabled(False)
        self.export_report_btn.setStyleSheet(self._get_export_button_style())
        
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_plots_btn)
        export_layout.addWidget(self.export_report_btn)
        export_layout.addStretch()
        
        layout.addWidget(export_frame)
    
    def _get_export_button_style(self) -> str:
        """获取导出按钮样式"""
        return """
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DA;
                border-radius: 6px;
                padding: 8px 16px;
                color: #24292E;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #F3F4F6;
                border-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #E3F2FD;
            }
            QPushButton:disabled {
                background-color: #F6F8FA;
                color: #A0AEC0;
                border-color: #E1E4E8;
            }
        """
    
    def _create_pareto_tab(self) -> QWidget:
        """创建Pareto前沿选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # 3D图画布
        self.pareto_figure = Figure(figsize=(10, 7), dpi=100)
        self.pareto_figure.patch.set_facecolor('#FFFFFF')
        self.pareto_canvas = FigureCanvas(self.pareto_figure)
        self.pareto_toolbar = NavigationToolbar(self.pareto_canvas, widget)
        
        layout.addWidget(self.pareto_toolbar)
        layout.addWidget(self.pareto_canvas, 1)
        
        # 视图切换按钮
        view_frame = QFrame()
        view_layout = QHBoxLayout(view_frame)
        view_layout.setContentsMargins(0, 8, 0, 0)
        
        view_label = QLabel("视图切换:")
        view_label.setStyleSheet("color: #5C6370;")
        view_layout.addWidget(view_label)
        
        self.view_3d_btn = QPushButton("3D视图")
        self.view_3d_btn.clicked.connect(lambda: self.update_pareto_view('3d'))
        self.view_3d_btn.setStyleSheet(self._get_view_button_style(True))
        
        self.view_2d_btn = QPushButton("2D投影")
        self.view_2d_btn.clicked.connect(lambda: self.update_pareto_view('2d'))
        self.view_2d_btn.setStyleSheet(self._get_view_button_style(False))
        
        view_layout.addWidget(self.view_3d_btn)
        view_layout.addWidget(self.view_2d_btn)
        view_layout.addStretch()
        
        layout.addWidget(view_frame)
        
        return widget
    
    def _get_view_button_style(self, active: bool) -> str:
        """获取视图按钮样式"""
        if active:
            return """
                QPushButton {
                    background-color: #1565C0;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: white;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #F6F8FA;
                    border: 1px solid #E1E4E8;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: #24292E;
                }
                QPushButton:hover {
                    background-color: #E3F2FD;
                    border-color: #1565C0;
                }
            """
    
    def _create_convergence_tab(self) -> QWidget:
        """创建收敛曲线选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        self.convergence_figure = Figure(figsize=(12, 8), dpi=100)
        self.convergence_figure.patch.set_facecolor('#FFFFFF')
        self.convergence_canvas = FigureCanvas(self.convergence_figure)
        self.convergence_toolbar = NavigationToolbar(self.convergence_canvas, widget)
        
        layout.addWidget(self.convergence_toolbar)
        layout.addWidget(self.convergence_canvas, 1)
        
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """创建数值结果选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # 统计摘要卡片
        summary_frame = QFrame()
        summary_frame.setStyleSheet("""
            QFrame {
                background-color: #F0F7FF;
                border: 1px solid #90CAF9;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        summary_layout = QVBoxLayout(summary_frame)
        
        summary_title = QLabel("优化结果摘要")
        summary_title.setFont(QFont("Microsoft YaHei UI", 11, QFont.Bold))
        summary_title.setStyleSheet("color: #1565C0; border: none; background: transparent;")
        summary_layout.addWidget(summary_title)
        
        self.summary_label = QLabel("等待优化运行...")
        self.summary_label.setWordWrap(True)
        self.summary_label.setFont(QFont("Consolas", 10))
        self.summary_label.setStyleSheet("color: #24292E; border: none; background: transparent;")
        summary_layout.addWidget(self.summary_label)
        
        layout.addWidget(summary_frame)
        
        # Pareto解表格
        table_frame = QFrame()
        table_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E1E4E8;
                border-radius: 8px;
            }
        """)
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(12, 12, 12, 12)
        
        table_title = QLabel("Pareto解集详情")
        table_title.setFont(QFont("Microsoft YaHei UI", 11, QFont.Bold))
        table_title.setStyleSheet("color: #24292E;")
        table_layout.addWidget(table_title)
        
        self.solutions_table = QTableWidget()
        self.solutions_table.setColumnCount(6)
        self.solutions_table.setHorizontalHeaderLabels([
            "编号", "Makespan (分钟)", "人工成本 (元)", "能耗 (kWh)", "Pareto排名", "拥挤度"
        ])
        self.solutions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.solutions_table.setAlternatingRowColors(True)
        self.solutions_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #E1E4E8;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #E3F2FD;
                color: #1565C0;
            }
            QHeaderView::section {
                background-color: #F6F8FA;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #E1E4E8;
                font-weight: bold;
                color: #24292E;
            }
        """)
        
        table_layout.addWidget(self.solutions_table)
        layout.addWidget(table_frame, 1)
        
        return widget
    
    def _create_log_tab(self) -> QWidget:
        """创建日志选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.log_text.clear)
        clear_btn.setStyleSheet(self._get_export_button_style())
        
        copy_btn = QPushButton("复制日志")
        copy_btn.clicked.connect(self._copy_log)
        copy_btn.setStyleSheet(self._get_export_button_style())
        
        btn_layout.addStretch()
        btn_layout.addWidget(copy_btn)
        btn_layout.addWidget(clear_btn)
        
        layout.addWidget(self.log_text, 1)
        layout.addLayout(btn_layout)
        
        return widget
    
    def _copy_log(self):
        """复制日志到剪贴板"""
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_text.toPlainText())
    
    def update_pareto_solutions(self, solutions: List[Solution], algorithm_name: str = "NSGA-II-VNS-MOSA"):
        """
        更新Pareto解集并刷新显示
        
        Args:
            solutions: Pareto解列表
            algorithm_name: 算法名称
        """
        self.pareto_solutions = solutions
        
        # 更新3D图
        self.pareto_figure.clear()
        if solutions:
            fig = plot_pareto_3d(solutions, title=f"{algorithm_name} Pareto前沿 ({len(solutions)}个解)")
            self._copy_figure(fig, self.pareto_figure)
            plt.close(fig)
        
        self.pareto_canvas.draw()
        self.current_figures['pareto'] = self.pareto_figure
        
        # 更新表格
        self._update_solutions_table(solutions)
        
        # 更新摘要
        self._update_summary(solutions)
        
        # 启用导出按钮
        self.export_csv_btn.setEnabled(bool(solutions))
        self.export_plots_btn.setEnabled(bool(solutions))
        self.export_report_btn.setEnabled(bool(solutions))
        
        # 更新视图按钮状态
        self.view_3d_btn.setStyleSheet(self._get_view_button_style(True))
        self.view_2d_btn.setStyleSheet(self._get_view_button_style(False))
    
    def update_pareto_view(self, view_type: str):
        """切换Pareto图视图类型"""
        self.pareto_figure.clear()
        
        if not self.pareto_solutions:
            return
        
        if view_type == '3d':
            fig = plot_pareto_3d(self.pareto_solutions)
            self.view_3d_btn.setStyleSheet(self._get_view_button_style(True))
            self.view_2d_btn.setStyleSheet(self._get_view_button_style(False))
        else:
            fig = plot_pareto_2d_projections(self.pareto_solutions)
            self.view_3d_btn.setStyleSheet(self._get_view_button_style(False))
            self.view_2d_btn.setStyleSheet(self._get_view_button_style(True))
        
        self._copy_figure(fig, self.pareto_figure)
        plt.close(fig)
        self.pareto_canvas.draw()
    
    def update_convergence(self, data_dict: Dict[str, Dict]):
        """
        更新收敛曲线
        
        Args:
            data_dict: {算法名: 收敛数据} 的字典
        """
        self.convergence_data = data_dict
        
        self.convergence_figure.clear()
        
        if data_dict:
            fig = plot_comparison(data_dict)
            self._copy_figure(fig, self.convergence_figure)
            plt.close(fig)
        
        self.convergence_canvas.draw()
        self.current_figures['convergence'] = self.convergence_figure
    
    def _copy_figure(self, source: Figure, target: Figure):
        """复制图形内容"""
        target.clear()
        
        for ax in source.axes:
            projection = ax.name if ax.name != 'rectilinear' else None
            new_ax = target.add_subplot(ax.get_subplotspec(), projection=projection)
            
            # 复制基本属性
            new_ax.set_title(ax.get_title(), fontsize=12, fontweight='bold')
            new_ax.set_xlabel(ax.get_xlabel(), fontsize=10)
            new_ax.set_ylabel(ax.get_ylabel(), fontsize=10)
            
            # 复制线条
            for line in ax.get_lines():
                new_ax.plot(line.get_xdata(), line.get_ydata(),
                           color=line.get_color(),
                           linewidth=line.get_linewidth(),
                           linestyle=line.get_linestyle(),
                           label=line.get_label())
            
            # 复制散点
            for collection in ax.collections:
                if hasattr(collection, 'get_offsets'):
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        colors = collection.get_facecolors()
                        if len(colors) > 0:
                            new_ax.scatter(offsets[:, 0], offsets[:, 1] if offsets.shape[1] > 1 else None,
                                          c=colors, alpha=0.7, s=50)
                        else:
                            new_ax.scatter(offsets[:, 0], offsets[:, 1] if offsets.shape[1] > 1 else None,
                                          alpha=0.7, s=50)
            
            if ax.get_legend():
                new_ax.legend(loc='best', fontsize=9)
            
            new_ax.grid(True, alpha=0.3, linestyle='--')
        
        target.tight_layout()
    
    def _update_solutions_table(self, solutions: List[Solution]):
        """更新解集表格"""
        self.solutions_table.setRowCount(len(solutions))
        
        for i, sol in enumerate(solutions):
            # 编号
            item0 = QTableWidgetItem(str(i + 1))
            item0.setTextAlignment(Qt.AlignCenter)
            self.solutions_table.setItem(i, 0, item0)
            
            # Makespan
            item1 = QTableWidgetItem(f"{sol.objectives[0]:.2f}")
            item1.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.solutions_table.setItem(i, 1, item1)
            
            # 人工成本
            item2 = QTableWidgetItem(f"{sol.objectives[1]:.2f}")
            item2.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.solutions_table.setItem(i, 2, item2)
            
            # 能耗
            item3 = QTableWidgetItem(f"{sol.objectives[2]:.2f}")
            item3.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.solutions_table.setItem(i, 3, item3)
            
            # 排名
            item4 = QTableWidgetItem(str(sol.rank))
            item4.setTextAlignment(Qt.AlignCenter)
            self.solutions_table.setItem(i, 4, item4)
            
            # 拥挤度
            cd_str = f"{sol.crowding_distance:.4f}" if sol.crowding_distance < float('inf') else "INF"
            item5 = QTableWidgetItem(cd_str)
            item5.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.solutions_table.setItem(i, 5, item5)
    
    def _update_summary(self, solutions: List[Solution]):
        """更新结果摘要"""
        if not solutions:
            self.summary_label.setText("无有效解")
            return
        
        objectives = np.array([s.objectives for s in solutions])
        
        summary = f"""优化完成! 共找到 {len(solutions)} 个Pareto最优解

目标函数统计:
{'─'*50}
  Makespan (F1) - 最大完工时间:
    最小值: {objectives[:, 0].min():.2f} 分钟
    最大值: {objectives[:, 0].max():.2f} 分钟
    平均值: {objectives[:, 0].mean():.2f} 分钟

  人工成本 (F2) - Labor Cost:
    最小值: {objectives[:, 1].min():.2f} 元
    最大值: {objectives[:, 1].max():.2f} 元
    平均值: {objectives[:, 1].mean():.2f} 元

  能耗 (F3) - Energy Consumption:
    最小值: {objectives[:, 2].min():.2f} kWh
    最大值: {objectives[:, 2].max():.2f} kWh
    平均值: {objectives[:, 2].mean():.2f} kWh
{'─'*50}
"""
        self.summary_label.setText(summary)
    
    def append_log(self, message: str):
        """追加日志消息"""
        # 根据消息类型添加颜色
        if "错误" in message or "Error" in message:
            colored_msg = f'<span style="color: #F44336;">{message}</span>'
        elif "警告" in message or "Warning" in message:
            colored_msg = f'<span style="color: #FF9800;">{message}</span>'
        elif "完成" in message or "成功" in message:
            colored_msg = f'<span style="color: #4CAF50;">{message}</span>'
        elif message.startswith("["):
            colored_msg = f'<span style="color: #64B5F6;">{message}</span>'
        else:
            colored_msg = f'<span style="color: #D4D4D4;">{message}</span>'
        
        self.log_text.append(colored_msg)
        
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def export_csv(self):
        """导出CSV文件"""
        if not self.pareto_solutions:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件", "pareto_solutions.csv", "CSV文件 (*.csv)"
        )
        
        if filepath:
            try:
                export_pareto_to_csv(self.pareto_solutions, filepath, include_decisions=True)
                QMessageBox.information(self, "导出成功", f"数据已保存到:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"保存失败: {str(e)}")
    
    def export_plots(self):
        """导出图表"""
        directory = QFileDialog.getExistingDirectory(self, "选择保存目录")
        
        if directory:
            try:
                # 保存Pareto图
                pareto_path = os.path.join(directory, "pareto_front.png")
                self.pareto_figure.savefig(pareto_path, dpi=200, bbox_inches='tight', 
                                          facecolor='white', edgecolor='none')
                
                # 保存收敛图
                convergence_path = os.path.join(directory, "convergence.png")
                self.convergence_figure.savefig(convergence_path, dpi=200, bbox_inches='tight',
                                               facecolor='white', edgecolor='none')
                
                QMessageBox.information(self, "导出成功", f"图表已保存到:\n{directory}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"保存失败: {str(e)}")
    
    def export_report(self):
        """导出完整报告"""
        directory = QFileDialog.getExistingDirectory(self, "选择报告保存目录")
        
        if directory:
            try:
                # 获取收敛数据
                conv_data = {}
                for key, val in self.convergence_data.items():
                    conv_data = val
                    break
                
                files = generate_report(
                    self.pareto_solutions,
                    conv_data,
                    'NSGA-II-VNS-MOSA',
                    directory
                )
                
                # 保存图表
                pareto_path = os.path.join(directory, "pareto_front.png")
                self.pareto_figure.savefig(pareto_path, dpi=200, bbox_inches='tight',
                                          facecolor='white', edgecolor='none')
                
                convergence_path = os.path.join(directory, "convergence.png")
                self.convergence_figure.savefig(convergence_path, dpi=200, bbox_inches='tight',
                                               facecolor='white', edgecolor='none')
                
                QMessageBox.information(self, "报告生成成功", 
                    f"报告已生成!\n\n保存位置: {directory}\n\n"
                    f"包含文件:\n"
                    f"- pareto_solutions.csv\n"
                    f"- summary.txt\n"
                    f"- pareto_front.png\n"
                    f"- convergence.png")
            except Exception as e:
                QMessageBox.critical(self, "报告生成失败", f"生成失败: {str(e)}")
    
    def clear(self):
        """清空所有结果"""
        self.pareto_solutions = []
        self.convergence_data = {}
        
        self.pareto_figure.clear()
        self.pareto_canvas.draw()
        
        self.convergence_figure.clear()
        self.convergence_canvas.draw()
        
        self.solutions_table.setRowCount(0)
        self.summary_label.setText("等待优化运行...")
        
        self.export_csv_btn.setEnabled(False)
        self.export_plots_btn.setEnabled(False)
        self.export_report_btn.setEnabled(False)
