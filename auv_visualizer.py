import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QOpenGLWidget, QVBoxLayout, 
                           QHBoxLayout, QWidget, QLabel, QFrame, QGridLayout, QPushButton, 
                           QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont

# 尝试导入OpenGL相关库
opengl_available = False
try:
    # 先检查基本的OpenGL导入
    from OpenGL import GL
    # 从GL模块导入基础函数
    from OpenGL.GL import (
        glBegin, glEnd, glVertex3f, glColor3f, glClear, glClearColor,
        glMatrixMode, glLoadIdentity, glRotatef, glTranslatef, 
        glViewport, glEnable, glPushMatrix, glPopMatrix, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_PROJECTION, GL_MODELVIEW, GL_LINES, GL_QUADS, GL_TRIANGLES, GL_DEPTH_TEST,
        GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN
    )
    # 从GLU模块导入透视和视角函数
    from OpenGL.GLU import gluLookAt, gluPerspective
    opengl_available = True
    print("✓ OpenGL库导入成功")
except ImportError as e:
    print(f"警告: 导入OpenGL时出错: {e}")
    print("将使用简单的替代显示方式")

class AUVVisualizer(QOpenGLWidget):
    def __init__(self, parent=None):
        if opengl_available:
            super().__init__(parent)
        else:
            super().__init__(parent)
            # 设置替代显示背景
            self.setStyleSheet("background-color: #1a1a2e;")
        
        self.parent = parent
        self.auv_model = None
        self.auv_controller = None
        
        # 相机参数
        self.camera_distance = 20.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 30.0
        self.camera_target = [0.0, 0.0, -5.0]
        
        # 场景参数
        self.show_grid = True
        self.show_water_surface = True
        self.show_axes = True
        self.show_thrust_vector = True
        
        # 物理模型的3D尺寸参数（调整为更细长的鱼雷形状）
        self.auv_length = 2.0  # 增加长度，使比例更像鱼雷
        self.auv_width = 0.25  # 减小宽度，使形状更细长
        self.auv_height = 0.25  # 保持与宽度一致，使横截面为圆形
        
        # 定时器，用于更新渲染
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)  # 约33FPS
        
        # 设置键盘跟踪
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 状态显示标签
        self.status_labels = {}
        self.setup_status_panel()
        
    def reset_view(self):
        """重置3D视角到初始状态"""
        self.camera_distance = 20.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 30.0
        self.camera_target = [0.0, 0.0, -5.0]
        self.update()  # 触发重绘
    
    def set_auv_model(self, model):
        """设置AUV模型"""
        self.auv_model = model
    
    def set_auv_controller(self, controller):
        """设置AUV控制器"""
        self.auv_controller = controller
    
    def setup_status_panel(self):
        """设置状态显示面板"""
        # 创建状态面板
        self.status_panel = QFrame(self.parent)
        self.status_panel.setFrameShape(QFrame.Panel)
        self.status_panel.setFrameShadow(QFrame.Raised)
        self.status_panel.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        
        # 创建网格布局
        self.status_layout = QGridLayout(self.status_panel)
        self.status_layout.setSpacing(5)
        
        # 添加状态标签
        status_items = [
            ('位置 (x,y,z):', 'position'),
            ('航向角:', 'heading'),
            ('俯仰角:', 'pitch'),
            ('横滚角:', 'roll'),
            ('推力:', 'thrust'),
            ('舵角1 (上垂直, z轴):', 'rudder1'),
            ('舵角2 (右水平, x轴):', 'rudder2'),
            ('舵角3 (下垂直, z轴):', 'rudder3'),
            ('舵角4 (左水平, x轴):', 'rudder4'),
            ('速度:', 'velocity')
        ]
        
        for i, (label_text, key) in enumerate(status_items):
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label = QLabel("0.0")
            value_label.setStyleSheet("font-weight: bold;")
            
            self.status_layout.addWidget(label, i, 0)
            self.status_layout.addWidget(value_label, i, 1)
            
            self.status_labels[key] = value_label
        
        # 添加舵角说明和控制提示
        info_text = "舵角说明:\n" \
                   "- 舵角基于AUV主体坐标系\n" \
                   "- 垂直舵: 绕z轴旋转 (控制左右转向)\n" \
                   "- 水平舵: 绕x轴旋转 (控制上下俯仰)\n" \
                   "- 角度单位: 度 (°)\n\n" \
                   "舵板位置、颜色和功能:\n" \
                   "- 上垂直舵 (红色): 绕z轴旋转，控制转向\n" \
                   "- 右水平舵 (绿色): 绕x轴旋转，控制俯仰\n" \
                   "- 下垂直舵 (蓝色): 绕z轴旋转，控制转向\n" \
                   "- 左水平舵 (黄色): 绕x轴旋转，控制俯仰\n\n" \
                   "转向控制策略:\n" \
                   "- 左转向: 红色舵板向左偏，蓝色舵板向右偏\n" \
                   "- 右转向: 红色舵板向右偏，蓝色舵板向左偏\n\n" \
                   "控制键位:\n" \
                   "W:前进 S:后退\n" \
                   "A:左转 D:右转\n" \
                   "↑↓:俯仰控制 →←:横滚控制\n" \
                   "Q:上浮 E:下潜\n\n" \
                   "独立舵板控制:\n" \
                   "1键: 红色舵板 (上垂直)\n" \
                   "2键: 绿色舵板 (右水平)\n" \
                   "3键: 蓝色舵板 (下垂直)\n" \
                   "4键: 黄色舵板 (左水平)\n" \
                   "Shift+1-4: 舵板反向控制\n\n" \
                   "鼠标操作:\n" \
                   "左键拖动: 旋转视角\n" \
                   "右键拖动: 平移视角\n" \
                   "滚轮: 缩放视角\n\n" \
                   "视角控制:\n" \
                   "R键: 复位视角到初始位置\n" \
                   "复位视角按钮: 复位3D视角"
        controls_label = QLabel(info_text)
        controls_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.status_layout.addWidget(controls_label, len(status_items), 0, 1, 2)
        
        self.status_panel.setLayout(self.status_layout)
        self.status_panel.setMinimumWidth(200)
    
    def initializeGL(self):
        """初始化OpenGL上下文"""
        if not opengl_available:
            return
        
        glClearColor(0.0, 0.3, 0.6, 1.0)  # 蓝色背景表示水
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 启用深度测试
        glEnable(GL_DEPTH_TEST)
    
    def resizeGL(self, w, h):
        """处理窗口大小变化"""
        if not opengl_available:
            return
        
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect_ratio = w / h if h != 0 else 1
        gluPerspective(45.0, aspect_ratio, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """渲染3D场景"""
        if not opengl_available:
            # 简单文本显示替代3D渲染
            self.update_status_labels()
            return
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # 设置相机位置
        cam_x = self.camera_distance * np.sin(np.radians(self.camera_azimuth)) * np.cos(np.radians(self.camera_elevation))
        cam_y = self.camera_distance * np.cos(np.radians(self.camera_azimuth)) * np.cos(np.radians(self.camera_elevation))
        cam_z = self.camera_distance * np.sin(np.radians(self.camera_elevation))
        
        gluLookAt(cam_x, cam_y, cam_z, 
                 self.camera_target[0], self.camera_target[1], self.camera_target[2],
                 0.0, 0.0, 1.0)
        
        # 绘制网格
        if self.show_grid:
            self.draw_grid()
        
        # 绘制水面
        if self.show_water_surface:
            self.draw_water_surface()
        
        # 绘制坐标轴
        # 坐标系始终显示，不受show_axes标志控制，确保用户能看到坐标方向
        self.draw_axes()
        
        # 绘制AUV模型
        if self.auv_model:
            self.draw_auv()
        
        # 更新状态标签
        self.update_status_labels()
    
    def draw_grid(self):
        """绘制水下网格"""
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_LINES)
        
        # 水平线
        for i in range(-10, 11):
            glVertex3f(-10.0, i, 0.0)
            glVertex3f(10.0, i, 0.0)
            glVertex3f(i, -10.0, 0.0)
            glVertex3f(i, 10.0, 0.0)
        
        glEnd()
    
    def draw_water_surface(self):
        # 绘制水面 - 半透明效果
        # 启用混合功能以实现半透明
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        
        # 使用glColor4f设置颜色和透明度(0.3表示30%不透明度)
        GL.glColor4f(0.0, 0.5, 0.8, 0.3)  # 半透明蓝色水面
        
        glBegin(GL_QUADS)
        glVertex3f(-20.0, -20.0, 0.0)
        glVertex3f(20.0, -20.0, 0.0)
        glVertex3f(20.0, 20.0, 0.0)
        glVertex3f(-20.0, 20.0, 0.0)
        glEnd()
        
        # 关闭混合功能，避免影响其他渲染
        GL.glDisable(GL.GL_BLEND)
    
    def draw_axes(self):
        """绘制带有箭头的三维坐标系"""
        # 保存当前矩阵状态
        glPushMatrix()
        
        # 坐标轴长度
        axis_length = 8.0
        
        # 绘制X轴 - 红色
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_length, 0.0, 0.0)
        glEnd()
        
        # X轴箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(axis_length, 0.0, 0.0)          # 箭头顶点
        glVertex3f(axis_length - 0.5, 0.2, 0.0)    # 箭头尾部点1
        glVertex3f(axis_length - 0.5, -0.2, 0.0)   # 箭头尾部点2
        glEnd()
        
        # 绘制Y轴 - 绿色
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_length, 0.0)
        glEnd()
        
        # Y轴箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, axis_length, 0.0)          # 箭头顶点
        glVertex3f(0.2, axis_length - 0.5, 0.0)    # 箭头尾部点1
        glVertex3f(-0.2, axis_length - 0.5, 0.0)   # 箭头尾部点2
        glEnd()
        
        # 绘制Z轴 - 蓝色
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_length)
        glEnd()
        
        # Z轴箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, 0.0, axis_length)          # 箭头顶点
        glVertex3f(0.2, 0.0, axis_length - 0.5)    # 箭头尾部点1
        glVertex3f(-0.2, 0.0, axis_length - 0.5)   # 箭头尾部点2
        glEnd()
        
        # 恢复矩阵状态
        glPopMatrix()
    
    def draw_auv(self):
        """绘制AUV模型（鱼雷形状）"""
        if not self.auv_model:
            return
        
        position = self.auv_model.get_position()
        orientation = self.auv_model.get_orientation()
        rudder_angles = self.auv_model.rudder_angles
        
        # 保存当前矩阵
        glPushMatrix()
        
        # 应用位置变换
        glTranslatef(position[0], position[1], position[2])
        
        # 应用旋转 (roll, pitch, yaw)
        glRotatef(np.degrees(orientation[0]), 1.0, 0.0, 0.0)  # roll
        glRotatef(np.degrees(orientation[1]), 0.0, 1.0, 0.0)  # pitch
        glRotatef(np.degrees(orientation[2]), 0.0, 0.0, 1.0)  # yaw
        
        # 调整比例使形状更像鱼雷（细长的圆柱+锥形）
        body_length = self.auv_length * 0.8  # 更长的主体
        body_radius = self.auv_width / 3  # 更细的主体
        
        # 绘制AUV主体（鱼雷形状）
        # 1. 绘制圆柱形中部
        self.draw_cylinder(body_length, body_radius)
        
        # 2. 绘制锥形前端（在正确的位置）
        glPushMatrix()
        glTranslatef(-(body_length / 2), 0, 0)  # 移动到圆柱体前端
        self.draw_cone(body_radius, self.auv_length * 0.2)  # 较短的锥形
        glPopMatrix()
        
        # 3. 绘制三维坐标系
        self.draw_axes()
        
        # 4. 绘制舵机（在正确的位置）
        self.draw_rudders(rudder_angles)
        
        # 5. 绘制推进器（在正确的位置）
        self.draw_propeller()
        
        # 如果启用推力矢量显示，且有推力，绘制推力矢量
        if self.show_thrust_vector and self.auv_model.thrust > 0:
            self.draw_thrust_vector()
        
        # 恢复矩阵
        glPopMatrix()
    
    def draw_small_axes(self, rotation_axis):
        """在旋转轴位置绘制小的三维坐标系，大小为0.3m，X方向与旋转轴对齐
        
        Args:
            rotation_axis: 旋转轴方向向量 (x, y, z)
        """
        # 保存当前矩阵状态
        glPushMatrix()
        
        # 坐标轴长度
        axis_length = 0.3
        arrow_length = 0.05  # 箭头长度
        arrow_width = 0.02   # 箭头宽度
        
        # 将X轴与旋转轴对齐
        # 计算从X轴到旋转轴的旋转矩阵
        x_axis = (1.0, 0.0, 0.0)
        
        # 如果旋转轴不是X轴，则计算旋转
        if np.array_equal(rotation_axis, x_axis) is False:
            # 计算旋转轴和角度
            rot_axis = np.cross(x_axis, rotation_axis)
            rot_angle = np.arccos(np.dot(x_axis, rotation_axis) / 
                                (np.linalg.norm(x_axis) * np.linalg.norm(rotation_axis)))
            
            # 应用旋转
            if np.linalg.norm(rot_axis) > 0.001:  # 避免除以零
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                glRotatef(np.degrees(rot_angle), rot_axis[0], rot_axis[1], rot_axis[2])
        
        # 绘制X轴 - 红色 (旋转轴方向)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_length, 0.0, 0.0)
        glEnd()
        
        # X轴箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(axis_length, 0.0, 0.0)                          # 箭头顶点
        glVertex3f(axis_length - arrow_length, arrow_width, 0.0)    # 箭头尾部点1
        glVertex3f(axis_length - arrow_length, -arrow_width, 0.0)   # 箭头尾部点2
        glEnd()
        
        # 绘制Y轴 - 绿色
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_length, 0.0)
        glEnd()
        
        # Y轴箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, axis_length, 0.0)                          # 箭头顶点
        glVertex3f(arrow_width, axis_length - arrow_length, 0.0)    # 箭头尾部点1
        glVertex3f(-arrow_width, axis_length - arrow_length, 0.0)   # 箭头尾部点2
        glEnd()
        
        # 绘制Z轴 - 蓝色
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_length)
        glEnd()
        
        # Z轴箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, 0.0, axis_length)                          # 箭头顶点
        glVertex3f(arrow_width, 0.0, axis_length - arrow_length)    # 箭头尾部点1
        glVertex3f(-arrow_width, 0.0, axis_length - arrow_length)   # 箭头尾部点2
        glEnd()
        
        # 恢复矩阵状态
        glPopMatrix()
        
    def draw_cylinder(self, length, radius):
        """绘制真正的圆柱体（使用三角形带）"""
        # 主体颜色 - 军用灰色鱼雷颜色
        glColor3f(0.2, 0.2, 0.25)
        
        # 参数设置
        slices = 16  # 圆周分段数
        
        # 圆柱体侧面（三角形带）
        glBegin(GL_TRIANGLE_STRIP)
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # 圆柱后端点
            glVertex3f(length/2, x, y)
            # 圆柱前端点
            glVertex3f(-length/2, x, y)
        glEnd()
        
        # 圆柱体前端盖
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(-length/2, 0, 0)  # 中心点
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(-length/2, x, y)
        glEnd()
        
        # 圆柱体后端盖
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(length/2, 0, 0)  # 中心点
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(length/2, x, y)
        glEnd()
    
    def draw_cone(self, radius, height):
        """绘制锥形前端"""
        # 锥形颜色 - 与主体一致的军用灰色
        glColor3f(0.2, 0.2, 0.25)
        
        # 参数设置
        slices = 16  # 圆周分段数
        
        # 绘制锥形侧面（三角形扇）
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(-height/2, 0, 0)  # 锥顶点
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(height/2, x, y)
        glEnd()
    
    def draw_rudders(self, angles):
        """绘制四个舵机（调整为鱼雷尾部的垂直和水平舵）"""
        # 舵机尺寸
        rudder_length = 0.5  # 进一步增加长度，更符合鱼雷尾部舵的比例
        rudder_width = 0.15  # 减小宽度，更细长
        rudder_thickness = 0.02
        
        # 舵机位置 (相对于AUV主体) - 放置在尾部
        radius = self.auv_width / 2
        positions = [
            (self.auv_length/2, 0, radius * 1.2),      # 上垂直舵
            (self.auv_length/2, radius * 1.2, 0),      # 右水平舵
            (self.auv_length/2, 0, -radius * 1.2),     # 下垂直舵
            (self.auv_length/2, -radius * 1.2, 0)      # 左水平舵
        ]
        
        # 四个舵机使用不同颜色以便区分
        rudder_colors = [
            (1.0, 0.0, 0.0),   # 舵板1 - 红色
            (0.0, 1.0, 0.0),   # 舵板2 - 绿色
            (0.0, 0.0, 1.0),   # 舵板3 - 蓝色
            (1.0, 1.0, 0.0)    # 舵板4 - 黄色
        ]
        
        # 旋转轴定义 - 相对于AUV本地坐标系的旋转轴
        # 由于draw_auv方法中已经应用了AUV的姿态变换，这里使用的坐标系已经是AUV的本地坐标系
        # 注意：垂直舵(0和2)在应用初始旋转后，旋转轴会改变
        rotation_axes = [
            (0.0, 0.0, 1.0),  # 上垂直舵 - 初始旋转前围绕AUV本地z轴旋转
            (0.0, 1.0, 0.0),  # 右水平舵 - 围绕AUV本地y轴旋转
            (0.0, 0.0, 1.0),  # 下垂直舵 - 初始旋转前围绕AUV本地z轴旋转
            (0.0, 1.0, 0.0)   # 左水平舵 - 围绕AUV本地y轴旋转
        ]
        
        # 舵机初始方向旋转 - 确保舵板在AUV坐标系中正确定向
        initial_rotations = [
            (90, 0, 0),    # 上垂直舵 - 绕X轴旋转90度，使短边平行于Z轴
            (0, 0, 0),     # 右水平舵 - 不需要额外旋转
            (90, 0, 0),    # 下垂直舵 - 绕X轴旋转90度，使短边平行于Z轴
            (0, 0, 0)      # 左水平舵 - 不需要额外旋转
        ]
        
        for i in range(4):
            x, y, z = positions[i]
            angle = angles[i]
            rot_x, rot_y, rot_z = initial_rotations[i]
            
            # 设置当前舵板的颜色
            glColor3f(*rudder_colors[i])
            
            glPushMatrix()
            # 移动到舵板位置
            glTranslatef(x, y, z)
            
            # 绘制舵板坐标系 - 大小0.3m，X方向为旋转轴方向
            # 对于垂直舵，使用Y轴作为旋转轴
            if i == 0 or i == 2:  # 上下垂直舵
                self.draw_small_axes((0, 1, 0))  # 使用Y轴作为旋转轴
            else:  # 左右水平舵
                self.draw_small_axes(rotation_axes[i])
            
            # 初始旋转 - 确保舵板在AUV坐标系中正确定向
            if rot_x != 0:
                glRotatef(rot_x, 1, 0, 0)
            if rot_y != 0:
                glRotatef(rot_y, 0, 1, 0)
            if rot_z != 0:
                glRotatef(rot_z, 0, 0, 1)
            
            # 应用舵角旋转 - 使用对应的旋转轴
            # 注意：由于draw_auv方法中已经应用了AUV的姿态变换，
            # 这里的旋转轴是相对于AUV本地坐标系的
            
            # 对于垂直舵，初始旋转改变了局部坐标系，所以需要使用新的旋转轴
            if i == 0 or i == 2:  # 上下垂直舵
                # 垂直舵在绕X轴旋转90度后，旋转轴变为Y轴(0, 1, 0)
                # 上垂直舵(0)：正转向角向左转，负转向角向右转
                # 下垂直舵(2)：正转向角向右转，负转向角向左转
                if i == 0:  # 上垂直舵
                    glRotatef(np.degrees(angle), 0, 1, 0)  # 使用新的旋转轴Y轴
                else:  # 下垂直舵
                    glRotatef(np.degrees(-angle), 0, 1, 0)  # 使用新的旋转轴Y轴
            else:  # 左右水平舵
                # 水平舵保持原旋转轴
                axis_x, axis_y, axis_z = rotation_axes[i]
                # 水平舵：正向旋转为向上，负向为向下
                glRotatef(np.degrees(angle), axis_x, axis_y, axis_z)
            
            # 绘制舵机
            l = rudder_length / 2
            w = rudder_width / 2
            t = rudder_thickness / 2
            
            glBegin(GL_QUADS)
            # 前面（靠近AUV主体的面）
            glVertex3f(0, -w, -t)
            glVertex3f(0, w, -t)
            glVertex3f(0, w, t)
            glVertex3f(0, -w, t)
            # 后面（远离AUV主体的面）
            glVertex3f(l, -w, -t)
            glVertex3f(l, w, -t)
            glVertex3f(l, w, t)
            glVertex3f(l, -w, t)
            # 顶面
            glVertex3f(0, -w, t)
            glVertex3f(0, w, t)
            glVertex3f(l, w, t)
            glVertex3f(l, -w, t)
            # 底面
            glVertex3f(0, -w, -t)
            glVertex3f(0, w, -t)
            glVertex3f(l, w, -t)
            glVertex3f(l, -w, -t)
            # 左侧面
            glVertex3f(0, -w, -t)
            glVertex3f(0, -w, t)
            glVertex3f(l, -w, t)
            glVertex3f(l, -w, -t)
            # 右侧面
            glVertex3f(0, w, -t)
            glVertex3f(0, w, t)
            glVertex3f(l, w, t)
            glVertex3f(l, w, -t)
            glEnd()
            
            glPopMatrix()
    
    def draw_propeller(self):
        """绘制推进器（鱼雷尾部螺旋桨）"""
        # 推进器颜色 - 金属银色
        glColor3f(0.5, 0.5, 0.5)
        
        # 移动到AUV尾部
        glPushMatrix()
        glTranslatef(self.auv_length/2 + 0.2, 0, 0)
        
        # 绘制推进器轴
        l = 0.15
        r = 0.03
        
        # 轴圆柱体
        slices = 8
        glBegin(GL_QUADS)
        for i in range(slices):
            angle1 = 2 * np.pi * i / slices
            angle2 = 2 * np.pi * (i + 1) / slices
            
            x1 = r * np.cos(angle1)
            y1 = r * np.sin(angle1)
            x2 = r * np.cos(angle2)
            y2 = r * np.sin(angle2)
            
            # 四个顶点组成一个四边形
            glVertex3f(-l/2, x1, y1)
            glVertex3f(-l/2, x2, y2)
            glVertex3f(l/2, x2, y2)
            glVertex3f(l/2, x1, y1)
        glEnd()
        
        # 绘制螺旋桨叶片（3个叶片，更像鱼雷的螺旋桨）
        blade_length = 0.3  # 增加叶片长度
        blade_width = 0.08  # 增加叶片宽度
        blade_angle = np.pi / 8  # 22.5度倾斜，更像实际螺旋桨
        
        for i in range(3):
            blade_rotation = 2 * np.pi * i / 3
            glPushMatrix()
            glRotatef(np.degrees(blade_rotation), 1, 0, 0)
            
            # 绘制叶片
            glBegin(GL_QUADS)
            glVertex3f(l/2, -blade_width/2, 0)
            glVertex3f(l/2, blade_width/2, 0)
            glVertex3f(l/2 + blade_length * np.cos(blade_angle), blade_width/2, blade_length * np.sin(blade_angle))
            glVertex3f(l/2 + blade_length * np.cos(blade_angle), -blade_width/2, blade_length * np.sin(blade_angle))
            glEnd()
            
            glPopMatrix()
        
        # 恢复初始矩阵
        glPopMatrix()
    
    def draw_thrust_vector(self):
        """绘制推力矢量"""
        if not self.auv_model or self.auv_model.thrust <= 0:
            return
        
        # 推力矢量颜色
        glColor3f(1.0, 0.5, 0.0)
        
        # 推力大小比例
        thrust_scale = self.auv_model.thrust / self.auv_model.propeller_max_thrust * 2.0
        
        # 起点在推进器后面
        start_x = self.auv_length/2 + 0.3
        
        glBegin(GL_LINES)
        glVertex3f(start_x, 0.0, 0.0)
        glVertex3f(start_x + thrust_scale, 0.0, 0.0)
        glEnd()
        
        # 绘制箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(start_x + thrust_scale, 0.0, 0.0)
        glVertex3f(start_x + thrust_scale - 0.1, -0.05, 0.0)
        glVertex3f(start_x + thrust_scale - 0.1, 0.05, 0.0)
        glEnd()
    
    def update_status_labels(self):
        """更新状态显示标签"""
        if not self.auv_model:
            return
        
        # 获取AUV状态
        position = self.auv_model.get_position()
        orientation = self.auv_model.get_orientation()
        velocity = self.auv_model.get_velocity()
        rudder_angles = self.auv_model.rudder_angles
        thrust = self.auv_model.thrust
        
        # 更新位置
        self.status_labels['position'].setText(f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        
        # 更新角度 (转换为度数)
        self.status_labels['heading'].setText(f"{np.degrees(orientation[2]):.2f}°")
        self.status_labels['pitch'].setText(f"{np.degrees(orientation[1]):.2f}°")
        self.status_labels['roll'].setText(f"{np.degrees(orientation[0]):.2f}°")
        
        # 更新推力
        self.status_labels['thrust'].setText(f"{thrust:.2f} N")
        
        # 更新舵角（带颜色编码以直观显示方向）
        for i in range(4):
            angle_deg = np.degrees(rudder_angles[i])
            # 根据角度正负设置不同颜色
            if angle_deg > 0:
                # 正角度显示为蓝色
                style = "font-weight: bold; color: blue;"
            elif angle_deg < 0:
                # 负角度显示为红色
                style = "font-weight: bold; color: red;"
            else:
                # 零角度显示为默认颜色
                style = "font-weight: bold;"
            
            self.status_labels[f'rudder{i+1}'].setStyleSheet(style)
            self.status_labels[f'rudder{i+1}'].setText(f"{angle_deg:.2f}°")
        
        # 更新速度
        speed = np.linalg.norm(velocity)
        self.status_labels['velocity'].setText(f"{speed:.2f} m/s")
    
    def keyPressEvent(self, event):
        """处理键盘按下事件"""
        # 检查视角重置快捷键
        if event.key() == Qt.Key_R:
            self.reset_view()
            return
            
        if self.auv_controller:
            key = event.key()
            
            # 处理按键映射
            key_map = {
                # 基本控制按键
                Qt.Key_W: 'w',
                Qt.Key_S: 's',
                Qt.Key_A: 'a',
                Qt.Key_D: 'd',
                Qt.Key_Up: 'up',
                Qt.Key_Down: 'down',
                Qt.Key_Left: 'left',
                Qt.Key_Right: 'right',
                Qt.Key_Q: 'q',
                Qt.Key_E: 'e',
                # 独立舵板控制按键
                Qt.Key_1: '1',
                Qt.Key_2: '2',
                Qt.Key_3: '3',
                Qt.Key_4: '4'
            }
            
            # 检查Shift键组合（用于舵板向下控制）
            if event.modifiers() & Qt.ShiftModifier:
                if key == Qt.Key_1:
                    self.auv_controller.set_key_state('!', True)
                    return
                elif key == Qt.Key_2:
                    self.auv_controller.set_key_state('@', True)
                    return
                elif key == Qt.Key_3:
                    self.auv_controller.set_key_state('#', True)
                    return
                elif key == Qt.Key_4:
                    self.auv_controller.set_key_state('$', True)
                    return
            
            if key in key_map:
                self.auv_controller.set_key_state(key_map[key], True)
        
        # 调用父类方法以确保Qt事件处理链完整
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """处理键盘释放事件"""
        if self.auv_controller:
            key = event.key()
            
            # 处理按键映射
            key_map = {
                # 基本控制按键
                Qt.Key_W: 'w',
                Qt.Key_S: 's',
                Qt.Key_A: 'a',
                Qt.Key_D: 'd',
                Qt.Key_Up: 'up',
                Qt.Key_Down: 'down',
                Qt.Key_Left: 'left',
                Qt.Key_Right: 'right',
                Qt.Key_Q: 'q',
                Qt.Key_E: 'e',
                # 独立舵板控制按键（向上）
                Qt.Key_1: '1',
                Qt.Key_2: '2',
                Qt.Key_3: '3',
                Qt.Key_4: '4'
            }
            
            # 处理普通按键释放
            if key in key_map:
                self.auv_controller.set_key_state(key_map[key], False)
            
            # 特殊处理Shift+1-4键释放
            # 只有当释放的是Shift键本身时，才释放所有对应的向下控制键
            if key == Qt.Key_Shift:
                self.auv_controller.set_key_state('!', False)  # Shift+1
                self.auv_controller.set_key_state('@', False)  # Shift+2
                self.auv_controller.set_key_state('#', False)  # Shift+3
                self.auv_controller.set_key_state('$', False)  # Shift+4
            # 当释放1-4键且没有按住Shift键时，释放对应的向下控制键
            elif key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4] and not (event.modifiers() & Qt.ShiftModifier):
                if key == Qt.Key_1:
                    self.auv_controller.set_key_state('!', False)  # Shift+1
                elif key == Qt.Key_2:
                    self.auv_controller.set_key_state('@', False)  # Shift+2
                elif key == Qt.Key_3:
                    self.auv_controller.set_key_state('#', False)  # Shift+3
                elif key == Qt.Key_4:
                    self.auv_controller.set_key_state('$', False)  # Shift+4
        
        # 调用父类方法以确保Qt事件处理链完整
        super().keyReleaseEvent(event)
    
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        self.last_mouse_pos = event.pos()
        self.last_mouse_button = event.button()
    
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if hasattr(self, 'last_mouse_pos') and hasattr(self, 'last_mouse_button'):
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            
            if self.last_mouse_button == Qt.LeftButton:
                # 左键拖动：旋转相机
                self.camera_azimuth += dx * 0.5
                self.camera_elevation -= dy * 0.5
                
                # 限制仰角范围
                self.camera_elevation = max(-89.0, min(89.0, self.camera_elevation))
            elif self.last_mouse_button == Qt.RightButton:
                # 右键拖动：平移相机
                # 计算平移速度
                pan_speed = 0.005 * self.camera_distance
                
                # 计算相机的前向和侧向向量
                # 转换到弧度
                az_rad = np.radians(self.camera_azimuth)
                el_rad = np.radians(self.camera_elevation)
                
                # 侧向向量 (水平方向)
                side_x = -np.sin(az_rad)
                side_y = np.cos(az_rad)
                side_z = 0.0
                
                # 上方向向量 (垂直方向)
                up_x = -np.sin(az_rad) * np.sin(el_rad)
                up_y = np.cos(az_rad) * np.sin(el_rad)
                up_z = np.cos(el_rad)
                
                # 根据鼠标移动更新相机目标位置
                self.camera_target[0] += (-dx * side_x * pan_speed + dy * up_x * pan_speed)
                self.camera_target[1] += (-dx * side_y * pan_speed + dy * up_y * pan_speed)
                self.camera_target[2] += (-dx * side_z * pan_speed + dy * up_z * pan_speed)
            
            self.last_mouse_pos = event.pos()
    
    def wheelEvent(self, event):
        """处理鼠标滚轮事件"""
        # 调整相机距离
        delta = event.angleDelta().y() / 120.0
        self.camera_distance = max(5.0, min(50.0, self.camera_distance - delta))

class AUVMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("AUV模拟器 - 单推进器四舵机控制")
        self.setGeometry(100, 100, 1000, 700)
        
        # 创建中心部件和布局
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # 创建3D可视化器
        self.visualizer = AUVVisualizer(self)
        
        # 创建控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 添加状态面板
        control_layout.addWidget(self.visualizer.status_panel, 1)
        
        # 添加复位按钮
        self.reset_button = QPushButton("复位AUV")
        self.reset_button.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; background-color: #4CAF50; color: white; border-radius: 4px; } QPushButton:hover { background-color: #45a049; }")
        self.reset_button.clicked.connect(self.handle_reset)
        control_layout.addWidget(self.reset_button)
        
        # 添加视角复位按钮
        self.reset_view_button = QPushButton("复位视角")
        self.reset_view_button.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; background-color: #2196F3; color: white; border-radius: 4px; } QPushButton:hover { background-color: #1976D2; }")
        self.reset_view_button.clicked.connect(self.visualizer.reset_view)
        control_layout.addWidget(self.reset_view_button)
        
        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)
        
        # 添加环境控制选项
        env_group = QGroupBox("环境设置")
        env_layout = QVBoxLayout()
        
        # 水流控制复选框
        self.current_checkbox = QCheckBox("启用水流")
        self.current_checkbox.setChecked(False)  # 默认禁用
        self.current_checkbox.stateChanged.connect(self.toggle_current)
        env_layout.addWidget(self.current_checkbox)
        
        # 波浪控制复选框
        self.waves_checkbox = QCheckBox("启用波浪")
        self.waves_checkbox.setChecked(False)  # 默认禁用
        self.waves_checkbox.stateChanged.connect(self.toggle_waves)
        env_layout.addWidget(self.waves_checkbox)
        
        env_group.setLayout(env_layout)
        control_layout.addWidget(env_group)
        
        # 添加到布局
        main_layout.addWidget(self.visualizer, 3)
        main_layout.addWidget(control_panel, 1)
        
        # 设置中心部件
        self.setCentralWidget(central_widget)
        
        # 保存AUVSimulation实例的引用
        self.simulation = None
    
    def set_simulation(self, simulation):
        """设置仿真系统引用"""
        self.simulation = simulation
    
    def handle_reset(self):
        """处理复位按钮点击事件"""
        if self.simulation:
            self.simulation.reset()
    
    def set_auv_model(self, model):
        """设置AUV模型"""
        self.visualizer.set_auv_model(model)
    
    def set_auv_controller(self, controller):
        """设置AUV控制器"""
        self.visualizer.set_auv_controller(controller)
    
    def toggle_current(self, state):
        """切换水流控制"""
        if self.simulation and self.simulation.auv_model:
            # 设置水流状态
            use_current = state == Qt.Checked
            self.simulation.auv_model.set_environment(use_current=use_current)
    
    def toggle_waves(self, state):
        """切换波浪控制"""
        if self.simulation and self.simulation.auv_model:
            # 设置波浪状态
            use_waves = state == Qt.Checked
            self.simulation.auv_model.set_environment(use_waves=use_waves)

# 检查依赖库是否安装
def check_dependencies():
    required_libraries = ['numpy', 'PyQt5']
    missing_libraries = []
    
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            missing_libraries.append(lib)
    
    # 尝试导入OpenGL
    try:
        from OpenGL import GL
        print("✓ OpenGL 模块可用")
    except ImportError:
        print("✗ OpenGL 模块不可用")
        print("警告: 未找到PyOpenGL库，3D渲染将不可用")
    
    if missing_libraries:
        print(f"错误: 缺少以下必需库: {', '.join(missing_libraries)}")
        print("请使用pip安装: pip install " + " ".join(missing_libraries))
        return False
    
    return True

# 主函数，用于直接运行可视化器
if __name__ == '__main__':
    if not check_dependencies():
        sys.exit(1)
    
    app = QApplication(sys.argv)
    window = AUVMainWindow()
    window.show()
    sys.exit(app.exec_())
