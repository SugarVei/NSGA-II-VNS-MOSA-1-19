"""
多目标调度优化系统
Multi-Objective Scheduling Optimization System

主程序入口
"""

import sys
import os
import multiprocessing

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import main

if __name__ == "__main__":
    # Windows 多进程必需：防止子进程重复执行主程序
    multiprocessing.freeze_support()
    main()
