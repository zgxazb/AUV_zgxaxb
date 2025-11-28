import numpy as np

class AUVModel:
    """AUV物理模型类，实现单推进器四舵机AUV的动力学仿真
    
    该类基于牛顿-欧拉方程实现AUV的运动模型，考虑了流体动力学效应、
    推进器和舵机的控制作用。
    """
    
    def __init__(self):
        """初始化AUV物理模型的基本参数"""
        # AUV基本参数
        self.length = 2.0  # 长度 (m)
        self.width = 0.5   # 宽度 (m)
        self.height = 0.5  # 高度 (m)
        self.mass = 200.0  # 质量 (kg)
        self.volume = self.length * self.width * self.height  # 体积 (m^3)
        
        # 惯性矩阵 (简化为对角矩阵)
        # 根据刚体转动惯量公式计算
        self.Ix = 1/12 * self.mass * (self.width**2 + self.height**2)  # 绕x轴转动惯量 (kg·m²)
        self.Iy = 1/12 * self.mass * (self.length**2 + self.height**2)  # 绕y轴转动惯量 (kg·m²)
        self.Iz = 1/12 * self.mass * (self.length**2 + self.width**2)  # 绕z轴转动惯量 (kg·m²)
        self.inertia_matrix = np.diag([self.Ix, self.Iy, self.Iz])  # 组合成对角惯性矩阵
        
        # 水动力学参数
        self.drag_coefficient_linear = np.array([10.0, 10.0, 15.0])  # 线性阻力系数 (N·s/m)
        self.drag_coefficient_angular = np.array([5.0, 5.0, 8.0])    # 角阻力系数 (N·m·s/rad)
        
        # 推进器参数
        self.propeller_max_thrust = 50.0  # 最大推力 (N)
        self.propeller_position = np.array([-self.length/2, 0.0, 0.0])  # 推进器位置 (m)
        
        # 舵机参数 (4个舵机: 2个水平舵，2个垂直舵)
        self.rudder_count = 4  # 舵机数量
        self.rudder_max_angle = np.pi/6  # 最大舵角 (30度)
        self.rudder_effectiveness = np.array([5.0, 5.0, 8.0, 8.0])  # 舵机效率系数 (N/rad)
        
        # 舵机位置 (在AUV本体坐标系中的坐标)
        # 位于尾部，90度均匀分布
        radius = min(self.width, self.height) / 3  # 舵机分布半径
        self.rudder_positions = np.array([
            [self.length/2 - 0.1, 0, radius],             # 上舵 (0度)
            [self.length/2 - 0.1, radius, 0],             # 右舵 (90度)
            [self.length/2 - 0.1, 0, -radius],            # 下舵 (180度)
            [self.length/2 - 0.1, -radius, 0]             # 左舵 (270度)
        ])
        
        # AUV状态向量 [x, y, z, phi, theta, psi, u, v, w, p, q, r]
        # x,y,z: 位置坐标 (m)
        # phi,theta,psi: 欧拉角 (横滚, 俯仰, 偏航) (rad)
        # u,v,w: 线速度 (前向, 侧向, 垂向) (m/s)
        # p,q,r: 角速度 (横滚, 俯仰, 偏航) (rad/s)
        self.state = np.zeros(12)
        
        # 控制输入 [propeller_thrust, rudder1_angle, rudder2_angle, rudder3_angle, rudder4_angle]
        # 推进器推力和四个舵机的角度
        self.control_input = np.zeros(5)
    
    @property
    def rudder_angles(self):
        """获取舵机角度"""
        return self.control_input[1:5]
    
    @property
    def thrust(self):
        """获取推进器推力"""
        return self.control_input[0]
        
    def update(self, dt):
        """更新AUV状态（与update_state方法相同，为了兼容性）"""
        self.update_state(dt)
        
    def update_state(self, dt):
        """使用简化的动力学模型更新AUV状态"""
        # 从状态向量中提取位置、姿态和速度
        x, y, z, phi, theta, psi = self.state[0:6]
        u, v, w, p, q, r = self.state[6:12]
        
        # 提取控制输入
        propeller_thrust = self.control_input[0]
        rudder_angles = self.control_input[1:5]
        
        # 计算流体动力
        linear_drag = -self.drag_coefficient_linear * np.array([u, v, w])
        angular_drag = -self.drag_coefficient_angular * np.array([p, q, r])
        
        # 计算推进力 (沿x轴正方向)
        thrust_force = np.array([propeller_thrust, 0.0, 0.0])
        
        # 计算舵机产生的力和力矩
        rudder_forces = np.zeros(3)
        rudder_torques = np.zeros(3)
        
        # 简化的舵机模型：根据舵角产生相应的力
        for i in range(self.rudder_count):
            angle = rudder_angles[i]
            effectiveness = self.rudder_effectiveness[i]
            
            # 计算舵机产生的力 (简化模型)
            rudder_force = effectiveness * angle * np.array([0.0, np.sin(angle), np.cos(angle)])
            
            # 根据舵机位置计算力矩
            torque = np.cross(self.rudder_positions[i], rudder_force)
            
            rudder_forces += rudder_force
            rudder_torques += torque
        
        # 总力和总力矩
        total_force = thrust_force + rudder_forces + linear_drag
        total_torque = rudder_torques + angular_drag
        
        # 计算加速度
        linear_acceleration = total_force / self.mass
        angular_acceleration = np.linalg.inv(self.inertia_matrix) @ total_torque
        
        # 更新速度
        u += linear_acceleration[0] * dt
        v += linear_acceleration[1] * dt
        w += linear_acceleration[2] * dt
        p += angular_acceleration[0] * dt
        q += angular_acceleration[1] * dt
        r += angular_acceleration[2] * dt
        
        # 更新位置和姿态 (使用当前姿态矩阵)
        # 简化的欧拉角更新
        rotation_matrix = self._euler_to_rotation_matrix(phi, theta, psi)
        velocity_vector = np.array([u, v, w])
        position_velocity = rotation_matrix @ velocity_vector
        
        x += position_velocity[0] * dt
        y += position_velocity[1] * dt
        z += position_velocity[2] * dt
        
        # 姿态角速度更新
        phi += p * dt
        theta += q * dt
        psi += r * dt
        
        # 规范化角度
        phi = self._normalize_angle(phi)
        theta = self._normalize_angle(theta)
        psi = self._normalize_angle(psi)
        
        # 更新状态向量
        self.state = np.array([x, y, z, phi, theta, psi, u, v, w, p, q, r])
    
    def _euler_to_rotation_matrix(self, phi, theta, psi):
        """将欧拉角转换为旋转矩阵"""
        # 横滚(phi)矩阵
        R_phi = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        # 俯仰(theta)矩阵
        R_theta = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        # 偏航(psi)矩阵
        R_psi = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵: R = R_psi * R_theta * R_phi
        rotation_matrix = R_psi @ R_theta @ R_phi
        return rotation_matrix
    
    def _normalize_angle(self, angle):
        """将角度规范化到 [-pi, pi] 范围内"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def set_control_input(self, thrust, rudder_angles):
        """设置控制输入，包括推进器推力和舵机角度"""
        # 限制推力范围，允许负值实现后退
        self.control_input[0] = np.clip(thrust, -self.propeller_max_thrust, self.propeller_max_thrust)
        
        # 限制舵角范围
        for i in range(self.rudder_count):
            self.control_input[i+1] = np.clip(rudder_angles[i], -self.rudder_max_angle, self.rudder_max_angle)
    
    def get_position(self):
        """获取AUV当前位置"""
        return self.state[0:3]
    
    def set_position(self, position):
        """设置AUV位置[x, y, z]"""
        self.state[0:3] = position
    
    def get_orientation(self):
        """获取AUV当前姿态角"""
        return self.state[3:6]
    
    def get_velocity(self):
        """获取AUV速度[u, v, w]（线速度）"""
        return self.state[6:9]
    
    def get_linear_velocity(self):
        """获取AUV线速度[u, v, w]"""
        return self.state[6:9]
    
    def get_angular_velocity(self):
        """获取AUV当前角速度"""
        return self.state[9:12]
    
    def set_velocity(self, velocity):
        """设置AUV速度[u, v, w, p, q, r]（线速度和角速度）
        
        参数:
        velocity: 长度为3（只设置线速度）或6（设置线速度和角速度）的数组
        """
        if len(velocity) == 3:
            # 只设置线速度，保持角速度不变
            self.state[6:9] = velocity
        elif len(velocity) == 6:
            # 设置完整的速度向量
            self.state[6:12] = velocity
        else:
            raise ValueError(f"速度数组长度必须为3或6，当前为{len(velocity)}")
    
    def reset(self):
        """重置AUV状态"""
        self.state = np.zeros(12)
        self.control_input = np.zeros(5)
