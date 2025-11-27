import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# 导入我们创建的模块
try:
    from auv_model import AUVModel
    from auv_controller import AUVController
    from auv_visualizer import AUVMainWindow, check_dependencies
    modules_available = True
except ImportError as e:
    print(f"错误: 无法导入必要的模块: {e}")
    modules_available = False

class AUVSimulation:
    def __init__(self):
        """初始化AUV仿真系统"""
        self.app = None
        self.window = None
        self.auv_model = None
        self.auv_controller = None
        self.simulation_timer = None
        
        # 仿真参数
        self.time_step = 0.01  # 10ms时间步长
        self.is_running = False
        self.real_time_factor = 1.0  # 实时因子，1.0表示实时仿真
        
    def initialize(self):
        """初始化仿真系统各组件"""
        print("正在初始化AUV仿真系统...")
        
        # 检查依赖
        if not check_dependencies():
            return False
        
        # 创建Qt应用程序
        self.app = QApplication(sys.argv)
        
        # 创建AUV物理模型
        self.auv_model = AUVModel()
        print("AUV物理模型初始化完成")
        
        # 创建控制器
        self.auv_controller = AUVController(self.auv_model)
        print("AUV控制器初始化完成")
        
        # 创建主窗口
        self.window = AUVMainWindow()
        self.window.set_auv_model(self.auv_model)
        self.window.set_auv_controller(self.auv_controller)
        self.window.set_simulation(self)  # 设置仿真系统引用
        print("可视化界面初始化完成")
        
        # 设置仿真定时器
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        
        print("仿真系统初始化完成")
        return True
    
    def start(self):
        """启动仿真系统"""
        if not self.initialize():
            print("仿真系统初始化失败，无法启动")
            return False
        
        # 显示主窗口
        self.window.show()
        
        # 开始仿真循环
        self.is_running = True
        self.simulation_timer.start(int(self.time_step * 1000 / self.real_time_factor))
        
        print("仿真系统已启动")
        print("\n控制键位说明：")
        print("W: 前进    S: 后退")
        print("A: 左转    D: 右转")
        print("↑: 上仰    ↓: 下俯")
        print("←: 左滚    →: 右滚")
        print("Q: 上浮    E: 下潜")
        print("\n舵板控制：")
        print("1-4键: 舵板向上")
        print("Shift+1-4: 舵板向下")
        print("\n鼠标操作：")
        print("拖动: 旋转视角    滚轮: 缩放视角")
        
        # 运行应用程序主循环
        sys.exit(self.app.exec_())
    
    def update_simulation(self):
        """更新仿真状态"""
        if not self.is_running:
            return
        
        try:
            # 更新控制器
            self.auv_controller.update(self.time_step)
            
            # 更新物理模型
            self.auv_model.update(self.time_step)
            
            # 检查和处理碰撞
            self._handle_collisions()
            
            # 限制AUV在合理范围内活动
            self._constrain_position()
            
        except Exception as e:
            print(f"仿真更新出错: {e}")
            self.stop()
    
    def _handle_collisions(self):
        """处理碰撞检测和响应"""
        position = self.auv_model.get_position()
        
        # 与水面碰撞检测 (z=0)
        if position[2] > 0:
            # 限制位置在水面以下
            self.auv_model.set_position([position[0], position[1], 0.0])
            
            # 消除z方向的速度
            velocity = list(self.auv_model.get_velocity())
            velocity[2] = 0.0
            self.auv_model.set_velocity(velocity)
            
            # 施加轻微的阻尼
            for i in range(3):
                velocity[i] *= 0.9
            self.auv_model.set_velocity(velocity)
        
        # 与海底碰撞检测 (假设海底在z=-50处)
        if position[2] < -50:
            # 限制位置在海底以上
            self.auv_model.set_position([position[0], position[1], -50.0])
            
            # 消除z方向的速度并反弹
            velocity = list(self.auv_model.get_velocity())
            velocity[2] = -velocity[2] * 0.3  # 反弹并损失能量
            self.auv_model.set_velocity(velocity)
    
    def _constrain_position(self):
        """限制AUV在合理范围内活动"""
        position = self.auv_model.get_position()
        max_range = 100.0
        
        # 检查x和y方向是否超出范围
        for i in range(2):
            if abs(position[i]) > max_range:
                # 如果超出范围，将AUV拉回边界内
                position[i] = np.sign(position[i]) * max_range
                
                # 消除该方向的速度
                velocity = list(self.auv_model.get_velocity())
                velocity[i] = 0.0
                self.auv_model.set_velocity(velocity)
        
        # 更新位置
        self.auv_model.set_position(position)
    
    def stop(self):
        """停止仿真系统"""
        print("正在停止仿真系统...")
        
        self.is_running = False
        
        if self.simulation_timer:
            self.simulation_timer.stop()
        
        if self.app:
            self.app.quit()
        
        print("仿真系统已停止")
    
    def reset(self):
        """重置仿真系统"""
        print("正在重置仿真系统...")
        
        # 暂停仿真
        was_running = self.is_running
        self.is_running = False
        
        if self.auv_model:
            self.auv_model.reset()
            print("AUV模型已重置")
        
        if self.auv_controller:
            self.auv_controller.reset()
            print("AUV控制器已重置")
        
        # 恢复仿真状态
        self.is_running = was_running
        print("仿真系统已重置")

def run_simulation():
    """运行AUV仿真的主函数"""
    print("====================================")
    print("        AUV仿真系统启动")
    print("   单推进器四舵机控制仿真")
    print("====================================")
    
    # 创建并启动仿真系统
    simulation = AUVSimulation()
    
    try:
        simulation.start()
    except KeyboardInterrupt:
        print("\n用户中断仿真")
    except Exception as e:
        print(f"仿真运行出错: {e}")
    finally:
        simulation.stop()
        print("仿真系统已完全关闭")

def check_module_availability():
    """检查所有必要模块是否可用"""
    modules = [
        ('numpy', 'numpy'),
        ('PyQt5', 'PyQt5'),
        ('OpenGL', 'OpenGL'),
        ('auv_model', 'auv_model'),
        ('auv_controller', 'auv_controller'),
        ('auv_visualizer', 'auv_visualizer')
    ]
    
    print("检查模块可用性...")
    all_available = True
    
    for name, module in modules:
        try:
            __import__(module)
            print(f"✓ {name} 模块可用")
        except ImportError as e:
            print(f"✗ {name} 模块不可用: {e}")
            all_available = False
    
    if all_available:
        print("所有模块检查通过")
    else:
        print("警告: 部分模块不可用，可能影响功能")
        print("建议安装缺失的Python库: pip install numpy PyQt5 PyOpenGL")
    
    return all_available

def display_welcome_message():
    """显示欢迎信息"""
    welcome = """
    =======================================================
          自主水下航行器 (AUV) 控制系统仿真器
    -------------------------------------------------------
    配置: 单推进器 + 四舵机
    功能: 3D可视化 + 实时控制 + 物理仿真
    =======================================================
    """
    print(welcome)

def main():
    """主入口函数"""
    # 显示欢迎信息
    display_welcome_message()
    
    # 检查模块可用性
    modules_ok = check_module_availability()
    
    if not modules_ok:
        user_input = input("\n尽管缺少部分模块，是否仍要尝试运行仿真？(y/n): ")
        if user_input.lower() != 'y':
            print("仿真已取消")
            return
    
    # 运行仿真
    print("\n正在启动仿真...")
    run_simulation()

# 如果作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
