"""
参数输入面板模块
Input Panel Module

提供问题参数和算法参数的输入界面，采用现代化设计风格。
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QSpinBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QPushButton, QComboBox,
    QScrollArea, QFrame, QSizePolicy, QCheckBox, QToolButton
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIcon


class CollapsibleGroupBox(QGroupBox):
    """可折叠的分组框"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self._is_collapsed = False
        self._content_widget = None
        self._animation = None
        
    def set_content_widget(self, widget: QWidget):
        """设置内容组件"""
        self._content_widget = widget
        
    def toggle_collapse(self):
        """切换折叠状态"""
        if self._content_widget:
            self._is_collapsed = not self._is_collapsed
            self._content_widget.setVisible(not self._is_collapsed)


class InputPanel(QWidget):
    """
    参数输入面板
    
    包含问题参数和算法参数的配置界面，采用卡片式布局。
    """
    
    # 信号: 参数变化时发出
    parameters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(16)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        
        # 1. 数据输入模式
        self.mode_group = self._create_mode_group()
        scroll_layout.addWidget(self.mode_group)
        
        # 2. 问题规模
        self.problem_group = self._create_problem_group()
        scroll_layout.addWidget(self.problem_group)
        
        # 3. 算法参数
        self.algorithm_group = self._create_algorithm_group()
        scroll_layout.addWidget(self.algorithm_group)
        
        # 4. 高级设置
        self.advanced_group = self._create_advanced_group()
        scroll_layout.addWidget(self.advanced_group)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    
    def _create_section_header(self, title: str, icon: str = "") -> QLabel:
        """创建区域标题"""
        label = QLabel(f"{icon} {title}" if icon else title)
        label.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        label.setStyleSheet("""
            QLabel {
                color: #1565C0;
                padding: 4px 0;
                border-bottom: 2px solid #E3F2FD;
                margin-bottom: 8px;
            }
        """)
        return label
    
    def _create_info_label(self, text: str, style: str = "info") -> QLabel:
        """创建信息提示标签"""
        label = QLabel(text)
        label.setWordWrap(True)
        
        styles = {
            "info": "color: #1565C0; background: #E3F2FD; border: 1px solid #90CAF9;",
            "warning": "color: #E65100; background: #FFF3E0; border: 1px solid #FFCC80;",
            "success": "color: #2E7D32; background: #E8F5E9; border: 1px solid #A5D6A7;",
            "error": "color: #C62828; background: #FFEBEE; border: 1px solid #EF9A9A;"
        }
        
        label.setStyleSheet(f"""
            QLabel {{
                {styles.get(style, styles['info'])}
                padding: 10px 12px;
                border-radius: 6px;
                font-size: 9pt;
            }}
        """)
        return label
    
    def _create_mode_group(self) -> QGroupBox:
        """创建数据输入模式选择组"""
        group = QGroupBox("数据输入模式")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)
        
        self.mode_button_group = QButtonGroup(self)
        
        # 模式选择区域
        mode_frame = QFrame()
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(16)
        
        # 自动生成模式卡片
        auto_card = self._create_mode_card(
            "自动生成",
            "系统自动生成符合逻辑的测试数据",
            True
        )
        self.auto_mode = auto_card.findChild(QRadioButton)
        mode_layout.addWidget(auto_card)
        
        # 手动输入模式卡片
        manual_card = self._create_mode_card(
            "手动输入",
            "自定义输入所有参数数据",
            False
        )
        self.manual_mode = manual_card.findChild(QRadioButton)
        mode_layout.addWidget(manual_card)
        
        self.mode_button_group.addButton(self.auto_mode, 0)
        self.mode_button_group.addButton(self.manual_mode, 1)
        self.mode_button_group.buttonClicked.connect(self._on_mode_changed)
        
        layout.addWidget(mode_frame)
        
        # 模式说明
        self.mode_description = self._create_info_label(
            "系统将自动生成符合逻辑的随机测试数据，适合快速验证算法效果",
            "info"
        )
        layout.addWidget(self.mode_description)
        
        # 手动输入按钮 (默认隐藏)
        self.manual_input_btn = QPushButton("打开数据输入界面")
        self.manual_input_btn.setProperty("secondary", True)
        self.manual_input_btn.setToolTip("点击输入加工时间、设置时间、能耗等详细数据")
        self.manual_input_btn.clicked.connect(self._open_manual_input_dialog)
        self.manual_input_btn.setVisible(False)
        layout.addWidget(self.manual_input_btn)
        
        # 手动输入状态标签
        self.manual_status_label = QLabel("")
        self.manual_status_label.setVisible(False)
        layout.addWidget(self.manual_status_label)
        
        # 随机种子设置
        self.seed_layout_widget = QFrame()
        seed_layout = QHBoxLayout(self.seed_layout_widget)
        seed_layout.setContentsMargins(0, 8, 0, 0)
        
        seed_label = QLabel("随机种子:")
        seed_label.setToolTip("设置随机种子以获得可重复的结果，0表示随机")
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.seed_spin.setSpecialValueText("随机")
        self.seed_spin.setMinimumWidth(100)
        
        seed_layout.addWidget(seed_label)
        seed_layout.addWidget(self.seed_spin)
        seed_layout.addStretch()
        
        layout.addWidget(self.seed_layout_widget)
        
        # 存储手动输入的数据
        self.manual_data = None
        
        return group
    
    def _create_mode_card(self, title: str, description: str, checked: bool) -> QFrame:
        """创建模式选择卡片"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 2px solid #E1E4E8;
                border-radius: 8px;
                padding: 8px;
            }
            QFrame:hover {
                border-color: #1E88E5;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(4)
        
        radio = QRadioButton(title)
        radio.setChecked(checked)
        radio.setFont(QFont("Microsoft YaHei UI", 9, QFont.Bold))
        
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #5C6370; font-size: 8pt;")
        desc_label.setWordWrap(True)
        
        layout.addWidget(radio)
        layout.addWidget(desc_label)
        
        return card
    
    def _on_mode_changed(self):
        """模式切换时的处理"""
        is_manual = self.manual_mode.isChecked()
        
        if is_manual:
            self.mode_description.setText(
                "手动输入模式：请点击下方按钮输入每个阶段每台机器的加工时间、设置时间、速度参数和能耗成本"
            )
            self.mode_description.setStyleSheet("""
                QLabel {
                    color: #E65100;
                    background: #FFF3E0;
                    border: 1px solid #FFCC80;
                    padding: 10px 12px;
                    border-radius: 6px;
                    font-size: 9pt;
                }
            """)
            self.manual_input_btn.setVisible(True)
            self.seed_layout_widget.setVisible(False)
            self._update_manual_status()
        else:
            self.mode_description.setText(
                "系统将自动生成符合逻辑的随机测试数据，适合快速验证算法效果"
            )
            self.mode_description.setStyleSheet("""
                QLabel {
                    color: #1565C0;
                    background: #E3F2FD;
                    border: 1px solid #90CAF9;
                    padding: 10px 12px;
                    border-radius: 6px;
                    font-size: 9pt;
                }
            """)
            self.manual_input_btn.setVisible(False)
            self.manual_status_label.setVisible(False)
            self.seed_layout_widget.setVisible(True)
    
    def _open_manual_input_dialog(self):
        """打开手动数据输入对话框"""
        from ui.manual_input_dialog import ManualDataInputDialog
        
        dialog = ManualDataInputDialog(
            n_jobs=self.n_jobs_spin.value(),
            n_stages=self.n_stages_spin.value(),
            machines_per_stage=self.machines_spin.value(),
            n_speed_levels=self.n_speeds_spin.value(),
            n_skill_levels=self.n_skills_spin.value(),
            parent=self
        )
        
        if dialog.exec_() == dialog.Accepted:
            self.manual_data = dialog.get_data()
            self._update_manual_status()
    
    def _update_manual_status(self):
        """更新手动输入状态显示"""
        if self.manual_data is not None:
            self.manual_status_label.setText("数据已输入完成")
            self.manual_status_label.setStyleSheet("""
                QLabel {
                    color: #2E7D32;
                    background: #E8F5E9;
                    border: 1px solid #A5D6A7;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-weight: bold;
                }
            """)
        else:
            self.manual_status_label.setText("尚未输入数据，请点击上方按钮")
            self.manual_status_label.setStyleSheet("""
                QLabel {
                    color: #E65100;
                    background: #FFF3E0;
                    border: 1px solid #FFCC80;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-weight: bold;
                }
            """)
        self.manual_status_label.setVisible(True)
    
    def _create_problem_group(self) -> QGroupBox:
        """创建问题规模设置组"""
        group = QGroupBox("问题规模")
        layout = QGridLayout(group)
        layout.setSpacing(12)
        layout.setColumnStretch(1, 1)
        
        params = [
            ("工件数量:", "n_jobs", 2, 100, 10, "需要调度的工件(Job)数量"),
            ("阶段数量:", "n_stages", 1, 20, 5, "生产过程的阶段(Stage)数"),
            ("每阶段机器数:", "machines", 1, 10, 3, "每个阶段可用的并行机器数量"),
            ("速度等级数:", "n_speeds", 1, 5, 3, "机器可运行的速度等级数 (低速/中速/高速)"),
            ("工人技能等级:", "n_skills", 1, 5, 3, "工人的技能划分等级数"),
        ]
        
        for row, (label_text, attr_name, min_val, max_val, default, tooltip) in enumerate(params):
            label = QLabel(label_text)
            label.setToolTip(tooltip)
            
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            spin.setMinimumWidth(80)
            setattr(self, f"{attr_name}_spin", spin)
            
            layout.addWidget(label, row, 0)
            layout.addWidget(spin, row, 1)
        
        return group
    
    def _create_algorithm_group(self) -> QGroupBox:
        """创建算法参数设置组"""
        group = QGroupBox("算法参数")
        main_layout = QVBoxLayout(group)
        main_layout.setSpacing(16)
        
        # NSGA-II 参数区域
        nsga_header = self._create_section_header("NSGA-II 参数", "")
        main_layout.addWidget(nsga_header)
        
        nsga_grid = QGridLayout()
        nsga_grid.setSpacing(10)
        nsga_grid.setColumnStretch(1, 1)
        nsga_grid.setColumnStretch(3, 1)
        
        # 种群大小
        nsga_grid.addWidget(QLabel("种群大小:"), 0, 0)
        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setRange(10, 500)
        self.pop_size_spin.setValue(200)
        self.pop_size_spin.setToolTip("每一代的解的数量，建议100-300")
        nsga_grid.addWidget(self.pop_size_spin, 0, 1)
        
        # 进化代数
        nsga_grid.addWidget(QLabel("进化代数:"), 0, 2)
        self.n_generations_spin = QSpinBox()
        self.n_generations_spin.setRange(10, 500)
        self.n_generations_spin.setValue(100)
        self.n_generations_spin.setToolTip("遗传算法的迭代次数，建议50-200")
        nsga_grid.addWidget(self.n_generations_spin, 0, 3)
        
        # 交叉概率
        nsga_grid.addWidget(QLabel("交叉概率:"), 1, 0)
        self.crossover_spin = QDoubleSpinBox()
        self.crossover_spin.setRange(0.5, 1.0)
        self.crossover_spin.setSingleStep(0.05)
        self.crossover_spin.setValue(0.95)
        self.crossover_spin.setToolTip("交叉操作的概率，建议0.8-0.95")
        nsga_grid.addWidget(self.crossover_spin, 1, 1)
        
        # 变异概率
        nsga_grid.addWidget(QLabel("变异概率:"), 1, 2)
        self.mutation_spin = QDoubleSpinBox()
        self.mutation_spin.setRange(0.01, 0.5)
        self.mutation_spin.setSingleStep(0.01)
        self.mutation_spin.setValue(0.15)
        self.mutation_spin.setToolTip("变异操作的概率，建议0.1-0.2")
        nsga_grid.addWidget(self.mutation_spin, 1, 3)
        
        main_layout.addLayout(nsga_grid)
        
        # 分隔线
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #E1E4E8;")
        line1.setFixedHeight(1)
        main_layout.addWidget(line1)
        
        # MOSA 参数区域
        mosa_header = self._create_section_header("MOSA 参数", "")
        main_layout.addWidget(mosa_header)
        
        mosa_grid = QGridLayout()
        mosa_grid.setSpacing(10)
        mosa_grid.setColumnStretch(1, 1)
        mosa_grid.setColumnStretch(3, 1)
        
        # 初始温度
        mosa_grid.addWidget(QLabel("初始温度:"), 0, 0)
        self.init_temp_spin = QDoubleSpinBox()
        self.init_temp_spin.setRange(10, 10000)
        self.init_temp_spin.setValue(1000)
        self.init_temp_spin.setToolTip("模拟退火的起始温度，建议500-2000")
        mosa_grid.addWidget(self.init_temp_spin, 0, 1)
        
        # 冷却系数
        mosa_grid.addWidget(QLabel("冷却系数:"), 0, 2)
        self.cooling_spin = QDoubleSpinBox()
        self.cooling_spin.setRange(0.80, 0.999)
        self.cooling_spin.setSingleStep(0.01)
        self.cooling_spin.setDecimals(3)
        self.cooling_spin.setValue(0.95)
        self.cooling_spin.setToolTip("温度衰减系数，建议0.90-0.98")
        mosa_grid.addWidget(self.cooling_spin, 0, 3)
        
        # 终止温度
        mosa_grid.addWidget(QLabel("终止温度:"), 1, 0)
        self.end_temp_spin = QDoubleSpinBox()
        self.end_temp_spin.setRange(0.001, 10)
        self.end_temp_spin.setDecimals(3)
        self.end_temp_spin.setValue(0.001)
        self.end_temp_spin.setToolTip("模拟退火的终止温度")
        mosa_grid.addWidget(self.end_temp_spin, 1, 1)
        
        # MOSA层数
        mosa_grid.addWidget(QLabel("MOSA层数:"), 1, 2)
        self.mosa_iterations_spin = QSpinBox()
        self.mosa_iterations_spin.setRange(10, 200)
        self.mosa_iterations_spin.setValue(50)
        self.mosa_iterations_spin.setToolTip("MOSA阶段的降温层数，建议30-100")
        mosa_grid.addWidget(self.mosa_iterations_spin, 1, 3)
        
        main_layout.addLayout(mosa_grid)
        
        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #E1E4E8;")
        line2.setFixedHeight(1)
        main_layout.addWidget(line2)
        
        # VNS 参数区域
        vns_header = self._create_section_header("VNS 参数", "")
        main_layout.addWidget(vns_header)
        
        vns_grid = QGridLayout()
        vns_grid.setSpacing(10)
        vns_grid.setColumnStretch(1, 1)
        vns_grid.setColumnStretch(3, 1)
        
        # VNS迭代次数
        vns_grid.addWidget(QLabel("局部搜索迭代:"), 0, 0)
        self.vns_iterations_spin = QSpinBox()
        self.vns_iterations_spin.setRange(1, 20)
        self.vns_iterations_spin.setValue(5)
        self.vns_iterations_spin.setToolTip("每个代表解每层生成候选的轮数")
        vns_grid.addWidget(self.vns_iterations_spin, 0, 1)
        
        # 邻居数量
        vns_grid.addWidget(QLabel("邻居采样数:"), 0, 2)
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(1, 10)
        self.neighbors_spin.setValue(3)
        self.neighbors_spin.setToolTip("每个邻域结构生成的邻居解数量")
        vns_grid.addWidget(self.neighbors_spin, 0, 3)
        
        main_layout.addLayout(vns_grid)
        
        return group
    
    def _create_advanced_group(self) -> QGroupBox:
        """创建高级设置组"""
        group = QGroupBox("高级设置")
        layout = QGridLayout(group)
        layout.setSpacing(12)
        layout.setColumnStretch(1, 1)
        
        # 代表解数量
        layout.addWidget(QLabel("代表解数量 (RP):"), 0, 0)
        self.n_representative_spin = QSpinBox()
        self.n_representative_spin.setRange(5, 100)
        self.n_representative_spin.setValue(40)
        self.n_representative_spin.setToolTip("MOSA中用于局部搜索的代表解数量")
        layout.addWidget(self.n_representative_spin, 0, 1)
        
        # 外部档案容量
        layout.addWidget(QLabel("档案容量 (AP):"), 1, 0)
        self.ap_size_spin = QSpinBox()
        self.ap_size_spin.setRange(50, 500)
        self.ap_size_spin.setValue(200)
        self.ap_size_spin.setToolTip("外部Pareto档案的最大容量")
        layout.addWidget(self.ap_size_spin, 1, 1)
        
        # epsilon-greedy参数
        layout.addWidget(QLabel("探索概率 (epsilon):"), 2, 0)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 0.5)
        self.epsilon_spin.setSingleStep(0.05)
        self.epsilon_spin.setValue(0.1)
        self.epsilon_spin.setToolTip("epsilon-greedy策略的探索概率")
        layout.addWidget(self.epsilon_spin, 2, 1)
        
        # 权重模式
        layout.addWidget(QLabel("权重模式:"), 3, 0)
        self.weight_mode_combo = QComboBox()
        self.weight_mode_combo.addItems(["随机权重 (推荐)", "固定权重"])
        self.weight_mode_combo.setToolTip("目标函数的权重分配方式")
        self.weight_mode_combo.currentIndexChanged.connect(self._on_weight_mode_changed)
        layout.addWidget(self.weight_mode_combo, 3, 1)
        
        # 固定权重设置 (默认隐藏)
        self.fixed_weights_widget = QFrame()
        weights_layout = QHBoxLayout(self.fixed_weights_widget)
        weights_layout.setContentsMargins(0, 0, 0, 0)
        
        weights_layout.addWidget(QLabel("F1:"))
        self.weight_f1_spin = QDoubleSpinBox()
        self.weight_f1_spin.setRange(0.0, 1.0)
        self.weight_f1_spin.setValue(0.33)
        self.weight_f1_spin.setSingleStep(0.1)
        self.weight_f1_spin.setDecimals(2)
        weights_layout.addWidget(self.weight_f1_spin)
        
        weights_layout.addWidget(QLabel("F2:"))
        self.weight_f2_spin = QDoubleSpinBox()
        self.weight_f2_spin.setRange(0.0, 1.0)
        self.weight_f2_spin.setValue(0.33)
        self.weight_f2_spin.setSingleStep(0.1)
        self.weight_f2_spin.setDecimals(2)
        weights_layout.addWidget(self.weight_f2_spin)
        
        weights_layout.addWidget(QLabel("F3:"))
        self.weight_f3_spin = QDoubleSpinBox()
        self.weight_f3_spin.setRange(0.0, 1.0)
        self.weight_f3_spin.setValue(0.34)
        self.weight_f3_spin.setSingleStep(0.1)
        self.weight_f3_spin.setDecimals(2)
        weights_layout.addWidget(self.weight_f3_spin)
        
        self.fixed_weights_widget.setVisible(False)
        layout.addWidget(self.fixed_weights_widget, 4, 0, 1, 2)
        
        # 审计模式
        self.audit_checkbox = QCheckBox("启用审计模式")
        self.audit_checkbox.setToolTip("记录详细的约束检查和VNS移动日志，用于论文验证")
        layout.addWidget(self.audit_checkbox, 5, 0, 1, 2)
        
        return group
    
    def _on_weight_mode_changed(self, index: int):
        """权重模式切换"""
        self.fixed_weights_widget.setVisible(index == 1)
    
    def get_parameters(self) -> dict:
        """
        获取所有参数值
        
        Returns:
            参数字典
        """
        weight_mode = "random" if self.weight_mode_combo.currentIndex() == 0 else "fixed"
        
        return {
            # 数据模式
            'auto_mode': self.auto_mode.isChecked(),
            'seed': self.seed_spin.value() if self.seed_spin.value() > 0 else None,
            'manual_data': self.manual_data,
            
            # 问题规模
            'n_jobs': self.n_jobs_spin.value(),
            'n_stages': self.n_stages_spin.value(),
            'machines_per_stage': self.machines_spin.value(),
            'n_speed_levels': self.n_speeds_spin.value(),
            'n_skill_levels': self.n_skills_spin.value(),
            
            # NSGA-II参数
            'pop_size': self.pop_size_spin.value(),
            'n_generations': self.n_generations_spin.value(),
            'crossover_prob': self.crossover_spin.value(),
            'mutation_prob': self.mutation_spin.value(),
            
            # MOSA参数
            'initial_temp': self.init_temp_spin.value(),
            'cooling_rate': self.cooling_spin.value(),
            'final_temp': self.end_temp_spin.value(),
            'mosa_layers': self.mosa_iterations_spin.value(),
            
            # VNS参数
            'vns_max_iters': self.vns_iterations_spin.value(),
            'neighbors_per_structure': self.neighbors_spin.value(),
            
            # 高级设置
            'rp_size': self.n_representative_spin.value(),
            'ap_size': self.ap_size_spin.value(),
            'epsilon_greedy': self.epsilon_spin.value(),
            'weight_mode': weight_mode,
            'fixed_weights': (
                self.weight_f1_spin.value(),
                self.weight_f2_spin.value(),
                self.weight_f3_spin.value()
            ),
            'audit_enabled': self.audit_checkbox.isChecked(),
        }
    
    def set_enabled(self, enabled: bool):
        """设置面板启用/禁用状态"""
        self.mode_group.setEnabled(enabled)
        self.problem_group.setEnabled(enabled)
        self.algorithm_group.setEnabled(enabled)
        self.advanced_group.setEnabled(enabled)
    
    def load_preset(self, preset_name: str):
        """
        加载预设参数配置
        
        Args:
            preset_name: 预设名称 ('small', 'medium', 'large', 'paper')
        """
        presets = {
            'small': {
                'n_jobs': 5, 'n_stages': 3, 'machines': 2,
                'pop_size': 50, 'n_generations': 50, 'mosa_layers': 30
            },
            'medium': {
                'n_jobs': 15, 'n_stages': 3, 'machines': 3,
                'pop_size': 100, 'n_generations': 100, 'mosa_layers': 50
            },
            'large': {
                'n_jobs': 30, 'n_stages': 5, 'machines': 4,
                'pop_size': 200, 'n_generations': 150, 'mosa_layers': 80
            },
            'paper': {
                'n_jobs': 15, 'n_stages': 3, 'machines': 3,
                'n_speeds': 3, 'n_skills': 3,
                'pop_size': 200, 'n_generations': 100,
                'crossover': 0.95, 'mutation': 0.15,
                'init_temp': 1000, 'cooling': 0.95, 'end_temp': 0.001,
                'mosa_layers': 50, 'rp_size': 40, 'ap_size': 200,
                'epsilon': 0.1, 'vns_iters': 5
            }
        }
        
        if preset_name in presets:
            p = presets[preset_name]
            if 'n_jobs' in p: self.n_jobs_spin.setValue(p['n_jobs'])
            if 'n_stages' in p: self.n_stages_spin.setValue(p['n_stages'])
            if 'machines' in p: self.machines_spin.setValue(p['machines'])
            if 'n_speeds' in p: self.n_speeds_spin.setValue(p['n_speeds'])
            if 'n_skills' in p: self.n_skills_spin.setValue(p['n_skills'])
            if 'pop_size' in p: self.pop_size_spin.setValue(p['pop_size'])
            if 'n_generations' in p: self.n_generations_spin.setValue(p['n_generations'])
            if 'crossover' in p: self.crossover_spin.setValue(p['crossover'])
            if 'mutation' in p: self.mutation_spin.setValue(p['mutation'])
            if 'init_temp' in p: self.init_temp_spin.setValue(p['init_temp'])
            if 'cooling' in p: self.cooling_spin.setValue(p['cooling'])
            if 'end_temp' in p: self.end_temp_spin.setValue(p['end_temp'])
            if 'mosa_layers' in p: self.mosa_iterations_spin.setValue(p['mosa_layers'])
            if 'rp_size' in p: self.n_representative_spin.setValue(p['rp_size'])
            if 'ap_size' in p: self.ap_size_spin.setValue(p['ap_size'])
            if 'epsilon' in p: self.epsilon_spin.setValue(p['epsilon'])
            if 'vns_iters' in p: self.vns_iterations_spin.setValue(p['vns_iters'])
