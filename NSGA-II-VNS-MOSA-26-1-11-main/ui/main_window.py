"""
主窗口模块
Main Window Module

整合所有UI组件，提供完整的用户界面。
采用现代化设计风格，支持NSGA-II-VNS-MOSA混合算法。
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QProgressBar, QLabel,
    QStatusBar, QMessageBox, QFrame, QApplication,
    QToolBar, QAction, QMenu, QMenuBar, QComboBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
import sys
import os
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.styles import MAIN_STYLESHEET, RUN_BUTTON_STYLE, STOP_BUTTON_STYLE, COLORS
from ui.input_panel import InputPanel
from ui.result_panel import ResultPanel

from models.problem import SchedulingProblem
from models.solution import Solution
from models.decoder import Decoder
from algorithms.nsga2_vns_mosa import NSGA2_VNS_MOSA


class OptimizationWorker(QThread):
    """
    优化算法工作线程
    
    在后台运行NSGA-II-VNS-MOSA混合算法，避免阻塞UI。
    """
    
    # 信号
    progress = pyqtSignal(int, int, str)  # current, total, message
    log = pyqtSignal(str)  # 日志消息
    phase1_finished = pyqtSignal(list, dict)  # NSGA-II阶段完成
    phase2_finished = pyqtSignal(list, dict)  # MOSA+VNS阶段完成
    error = pyqtSignal(str)  # 错误消息
    finished = pyqtSignal()  # 完成信号
    
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._is_cancelled = False
        self.problem = None
        self.algorithm = None
    
    def run(self):
        """运行优化"""
        try:
            params = self.params
            
            # 1. 创建问题实例
            self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 创建调度问题实例...")
            self.progress.emit(0, 100, "初始化问题...")
            
            if params['auto_mode']:
                machines_per_stage = [params['machines_per_stage']] * params['n_stages']
                self.problem = SchedulingProblem.generate_random(
                    n_jobs=params['n_jobs'],
                    n_stages=params['n_stages'],
                    machines_per_stage=machines_per_stage,
                    n_speed_levels=params['n_speed_levels'],
                    n_skill_levels=params['n_skill_levels'],
                    seed=params['seed']
                )
            else:
                # 手动输入模式
                manual_data = params.get('manual_data')
                machines_per_stage = [params['machines_per_stage']] * params['n_stages']
                
                if manual_data is not None:
                    self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 使用手动输入的数据...")
                    import numpy as np
                    
                    base_proc_time = manual_data['processing_time']
                    n_jobs, n_stages, n_machines = base_proc_time.shape
                    n_speeds = params['n_speed_levels']
                    
                    processing_time = np.zeros((n_jobs, n_stages, n_machines, n_speeds))
                    for job in range(n_jobs):
                        for stage in range(n_stages):
                            for machine in range(n_machines):
                                base_time = base_proc_time[job, stage, machine]
                                for speed in range(n_speeds):
                                    speed_factor = 1.0 - 0.25 * speed
                                    processing_time[job, stage, machine, speed] = base_time * speed_factor
                    
                    energy_rate = manual_data['energy_rate']
                    skill_wages = manual_data['skill_wages']
                    workers_available = manual_data['workers_available']
                    skill_compatibility = np.array([i for i in range(params['n_skill_levels'])])
                    
                    self.problem = SchedulingProblem(
                        n_jobs=params['n_jobs'],
                        n_stages=params['n_stages'],
                        machines_per_stage=machines_per_stage,
                        n_speed_levels=params['n_speed_levels'],
                        n_skill_levels=params['n_skill_levels'],
                        processing_time=processing_time,
                        energy_rate=energy_rate,
                        skill_wages=skill_wages,
                        skill_compatibility=skill_compatibility,
                        workers_available=workers_available
                    )
                else:
                    self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 警告: 手动模式但未输入数据，使用随机生成...")
                    self.problem = SchedulingProblem.generate_random(
                        n_jobs=params['n_jobs'],
                        n_stages=params['n_stages'],
                        machines_per_stage=machines_per_stage,
                        n_speed_levels=params['n_speed_levels'],
                        n_skill_levels=params['n_skill_levels'],
                        seed=params['seed']
                    )
            
            self.log.emit(self.problem.summary())
            
            if self._is_cancelled:
                return
            
            # 2. 创建并运行NSGA-II-VNS-MOSA混合算法
            self.log.emit(f"\n[{datetime.now().strftime('%H:%M:%S')}] 启动NSGA-II-VNS-MOSA混合算法...")
            
            self.algorithm = NSGA2_VNS_MOSA(
                problem=self.problem,
                pop_size=params['pop_size'],
                n_generations=params['n_generations'],
                crossover_prob=params['crossover_prob'],
                mutation_prob=params['mutation_prob'],
                initial_temp=params['initial_temp'],
                cooling_rate=params['cooling_rate'],
                final_temp=params['final_temp'],
                mosa_layers=params['mosa_layers'],
                rp_size=params['rp_size'],
                ap_size=params['ap_size'],
                epsilon_greedy=params['epsilon_greedy'],
                vns_max_iters=params['vns_max_iters'],
                weight_mode=params['weight_mode'],
                fixed_weights=params['fixed_weights'],
                audit_enabled=params.get('audit_enabled', False),
                seed=params['seed']
            )
            
            def progress_callback(current, total, msg):
                if self._is_cancelled:
                    return
                progress = int(current / total * 100)
                self.progress.emit(progress, 100, msg)
                if "Phase 1" in msg or "Phase 2" in msg:
                    self.log.emit(f"  {msg}")
            
            self.algorithm.set_progress_callback(progress_callback)
            
            # 运行算法
            pareto_archive = self.algorithm.run()
            
            self.log.emit(f"\n[{datetime.now().strftime('%H:%M:%S')}] 算法完成，Pareto档案大小: {len(pareto_archive)}")
            
            # 发送结果
            convergence_data = self.algorithm.get_convergence_data()
            self.phase2_finished.emit(pareto_archive, convergence_data)
            
            self.progress.emit(100, 100, "优化完成！")
            self.log.emit(f"\n[{datetime.now().strftime('%H:%M:%S')}] 优化流程完成!")
            
        except Exception as e:
            import traceback
            self.error.emit(f"优化过程出错: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()
    
    def cancel(self):
        """取消优化"""
        self._is_cancelled = True


class MainWindow(QMainWindow):
    """
    主窗口
    
    整合参数输入、优化运行和结果展示，采用现代化设计风格。
    """
    
    def __init__(self):
        super().__init__()
        
        self.worker: OptimizationWorker = None
        self.convergence_data = {}
        self.setup_ui()
        self.apply_styles()
        self.create_menu_bar()
    
    def setup_ui(self):
        """初始化UI"""
        self.setWindowTitle("NSGA-II-VNS-MOSA 多目标调度优化系统")
        self.setMinimumSize(1400, 900)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)
        
        # 顶部工具栏区域
        toolbar_frame = self._create_toolbar_frame()
        main_layout.addWidget(toolbar_frame)
        
        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #E1E4E8;
            }
            QSplitter::handle:hover {
                background-color: #1565C0;
            }
        """)
        
        # 左侧: 输入面板
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(12)
        
        self.input_panel = InputPanel()
        left_layout.addWidget(self.input_panel)
        
        # 按钮区域
        button_frame = QFrame()
        button_layout = QVBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        
        # 运行按钮
        self.run_button = QPushButton("运行优化")
        self.run_button.setStyleSheet(RUN_BUTTON_STYLE)
        self.run_button.clicked.connect(self.start_optimization)
        self.run_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(self.run_button)
        
        # 取消按钮
        self.cancel_button = QPushButton("停止")
        self.cancel_button.setStyleSheet(STOP_BUTTON_STYLE)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_optimization)
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(self.cancel_button)
        
        left_layout.addWidget(button_frame)
        
        left_widget.setMaximumWidth(420)
        left_widget.setMinimumWidth(380)
        splitter.addWidget(left_widget)
        
        # 右侧: 结果面板
        self.result_panel = ResultPanel()
        splitter.addWidget(self.result_panel)
        
        # 设置分割比例
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter, 1)
        
        # 底部进度条区域
        progress_frame = self._create_progress_frame()
        main_layout.addWidget(progress_frame)
        
        # 状态栏
        self.statusBar().showMessage("就绪 | NSGA-II-VNS-MOSA 多目标调度优化系统")
    
    def _create_toolbar_frame(self) -> QFrame:
        """创建顶部工具栏"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)
        
        # 标题
        title_label = QLabel("NSGA-II-VNS-MOSA")
        title_label.setFont(QFont("Microsoft YaHei UI", 14, QFont.Bold))
        title_label.setStyleSheet(f"color: {COLORS['primary']};")
        layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel("混合流水车间多目标调度优化")
        subtitle_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10pt;")
        layout.addWidget(subtitle_label)
        
        layout.addStretch()
        
        # 预设选择
        preset_label = QLabel("快速预设:")
        preset_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(preset_label)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["自定义", "小规模测试", "中等规模", "大规模", "论文参数"])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.preset_combo.setMinimumWidth(120)
        layout.addWidget(self.preset_combo)
        
        return frame
    
    def _create_progress_frame(self) -> QFrame:
        """创建底部进度条区域"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        
        # 状态图标
        self.status_icon = QLabel("")
        self.status_icon.setFixedWidth(24)
        layout.addWidget(self.status_icon)
        
        # 进度标签
        self.progress_label = QLabel("就绪")
        self.progress_label.setMinimumWidth(200)
        self.progress_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 500;")
        layout.addWidget(self.progress_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(24)
        layout.addWidget(self.progress_bar, 1)
        
        # 时间显示
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 9pt;")
        self.time_label.setMinimumWidth(100)
        layout.addWidget(self.time_label)
        
        return frame
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        export_action = QAction("导出结果...", self)
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _on_preset_changed(self, preset_name: str):
        """预设切换"""
        preset_map = {
            "小规模测试": "small",
            "中等规模": "medium",
            "大规模": "large",
            "论文参数": "paper"
        }
        
        if preset_name in preset_map:
            self.input_panel.load_preset(preset_map[preset_name])
    
    def _export_results(self):
        """导出结果"""
        QMessageBox.information(self, "导出", "导出功能开发中...")
    
    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于",
            "<h3>NSGA-II-VNS-MOSA 多目标调度优化系统</h3>"
            "<p>版本: 1.0.0</p>"
            "<p>基于论文实现的混合流水车间多目标调度优化算法</p>"
            "<p>三个优化目标:</p>"
            "<ul>"
            "<li>F1: 最小化最大完工时间 (Makespan)</li>"
            "<li>F2: 最小化人工成本 (Labor Cost)</li>"
            "<li>F3: 最小化能源消耗 (Energy Consumption)</li>"
            "</ul>"
        )
    
    def apply_styles(self):
        """应用样式"""
        self.setStyleSheet(MAIN_STYLESHEET)
    
    def start_optimization(self):
        """开始优化"""
        # 获取参数
        params = self.input_panel.get_parameters()
        
        # 验证参数
        if params['n_jobs'] < 2:
            QMessageBox.warning(self, "参数错误", "工件数量至少为2")
            return
        
        # 检查手动输入模式是否已输入数据
        if not params['auto_mode'] and params.get('manual_data') is None:
            reply = QMessageBox.question(
                self, "数据未输入",
                "手动输入模式下尚未输入数据。\n\n是否继续使用随机生成的数据？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # 清空之前的结果
        self.result_panel.clear()
        self.convergence_data = {}
        
        # 禁用输入
        self.input_panel.set_enabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.preset_combo.setEnabled(False)
        
        # 更新状态
        self.progress_label.setText("正在初始化...")
        self.progress_bar.setValue(0)
        self.start_time = datetime.now()
        
        # 创建并启动工作线程
        self.worker = OptimizationWorker(params)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.result_panel.append_log)
        self.worker.phase2_finished.connect(self.on_optimization_finished)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        
        self.result_panel.append_log(f"[{datetime.now().strftime('%H:%M:%S')}] 开始优化...")
        self.result_panel.append_log(f"问题规模: {params['n_jobs']}工件 x {params['n_stages']}阶段 x {params['machines_per_stage']}机器")
        self.result_panel.append_log(f"算法参数: 种群={params['pop_size']}, 代数={params['n_generations']}, MOSA层数={params['mosa_layers']}")
        
        self.worker.start()
    
    def cancel_optimization(self):
        """取消优化"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.result_panel.append_log("正在取消优化...")
            self.statusBar().showMessage("正在取消...")
    
    def on_progress(self, current: int, total: int, message: str):
        """进度更新"""
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)
        
        # 更新运行时间
        if hasattr(self, 'start_time'):
            elapsed = datetime.now() - self.start_time
            self.time_label.setText(f"已运行: {elapsed.seconds // 60}分{elapsed.seconds % 60}秒")
        
        self.statusBar().showMessage(message)
    
    def on_optimization_finished(self, pareto_solutions: list, convergence_data: dict):
        """优化完成"""
        self.convergence_data = convergence_data
        
        # 更新结果面板
        self.result_panel.update_pareto_solutions(pareto_solutions, "NSGA-II-VNS-MOSA")
        self.result_panel.update_convergence({"NSGA-II-VNS-MOSA": convergence_data})
        
        self.statusBar().showMessage(f"优化完成，找到{len(pareto_solutions)}个Pareto解")
    
    def on_error(self, error_message: str):
        """错误处理"""
        QMessageBox.critical(self, "优化错误", error_message)
        self.result_panel.append_log(f"错误: {error_message}")
    
    def on_finished(self):
        """优化完成"""
        self.input_panel.set_enabled(True)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.preset_combo.setEnabled(True)
        
        if self.progress_bar.value() >= 100:
            self.statusBar().showMessage("优化完成")
            self.progress_label.setText("优化完成")
        else:
            self.statusBar().showMessage("优化已取消")
            self.progress_label.setText("已取消")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "确认退出",
                "优化正在运行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """程序入口"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用程序调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS['background']))
    palette.setColor(QPalette.WindowText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.Base, QColor(COLORS['surface']))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS['surface_alt']))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS['text_inverse']))
    palette.setColor(QPalette.Text, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.Button, QColor(COLORS['surface']))
    palette.setColor(QPalette.ButtonText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.Highlight, QColor(COLORS['primary']))
    palette.setColor(QPalette.HighlightedText, QColor(COLORS['text_inverse']))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
