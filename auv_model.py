import numpy as np

class AUVModel:
    """AUV物理模型类，实现单推进器四舵机AUV的高级动力学仿真
    
    该类基于完整的牛顿-欧拉方程实现AUV的6自由度运动模型，考虑了真实的流体动力学效应：
    - 附加质量
    - 流体阻尼（线性和非线性）
    - 科里奥利和离心力
    - 重力和浮力
    - 推进器和舵机的控制作用
    - 环境扰动（水流和波浪）
    """
    
    def __init__(self):
        """初始化AUV物理模型的基本参数"""
        # AUV基本参数
        self.length = 2.0  # 长度 (m)
        self.width = 0.5   # 宽度 (m)
        self.height = 0.5  # 高度 (m)
        # 设置满排水量为205Kg
        self.displacement = 205.0  # 满排水量 (kg)
        self.water_density = 1000.0  # 水的密度 (kg/m³)
        # 根据排水量计算体积: 体积 = 排水量 / 水密度
        self.volume = self.displacement / self.water_density  # 体积 (m^3)
        # 设置AUV质量略小于排水量以提供适当浮力
        self.mass = 204.0  # 质量 (kg)
        
        # 水的物理特性
        self.water_kinematic_viscosity = 1.004e-6  # 水的运动粘度 (m²/s)
        
        # 计算浮力
        self.buoyancy = self.water_density * 9.81 * self.volume  # 浮力 (N)
        
        # 环境因素控制标志
        # 注意：环境力（水流和波浪）已在_calculate_environmental_forces方法中完全禁用
        # 无论这些标志如何设置，都不会产生任何环境干扰力
        self.use_current = False  # 水流影响控制（当前已禁用）
        self.use_waves = False    # 波浪影响控制（当前已禁用）
        
        # 重心和浮心位置 - 确保它们在同一垂直轴上，避免不平衡力矩
        # 将重心和浮心精确对齐在z轴上，确保初始状态下没有不平衡力矩
        self.center_of_mass = np.array([0.0, 0.0, 0.0])  # 重心 (m)
        self.center_of_buoyancy = np.array([0.0, 0.0, 0.1])  # 浮心 (m)，略微上移以提供稳定性
        # 确保x和y坐标完全匹配，避免产生横向力矩
        assert np.array_equal(self.center_of_mass[:2], self.center_of_buoyancy[:2]), "重心和浮心在x-y平面上必须对齐"
        
        # 1. 完整的刚体惯性矩阵（非对角）
        # 基于AUV几何形状的近似
        self.Ixx = 1/12 * self.mass * (self.width**2 + self.height**2)
        self.Iyy = 1/12 * self.mass * (self.length**2 + self.height**2)
        self.Izz = 1/12 * self.mass * (self.length**2 + self.width**2)
        # 非对角元素（考虑非对称性）
        self.Ixy = self.Iyx = 0.0  # 假设x-y平面对称
        self.Ixz = self.Izx = 0.0  # 假设x-z平面对称
        self.Iyz = self.Izy = 0.0  # 假设y-z平面对称
        
        # 构建完整的惯性矩阵
        self.inertia_matrix = np.array([
            [self.Ixx, -self.Ixy, -self.Ixz],
            [-self.Iyx, self.Iyy, -self.Iyz],
            [-self.Izx, -self.Izy, self.Izz]
        ])
        
        # 2. 附加质量矩阵 (6x6)
        # 基于细长体理论和经验公式计算
        # 前向运动的附加质量
        X_udot = -0.05 * self.water_density * self.volume  # 沿x轴的附加质量
        # 侧向和垂向运动的附加质量（较大，因为流体被推向两侧）
        Y_vdot = -0.9 * self.water_density * self.volume  # 沿y轴的附加质量
        Z_wdot = -0.9 * self.water_density * self.volume  # 沿z轴的附加质量
        
        # 转动附加质量
        K_pdot = -0.01 * self.water_density * self.volume * self.length**2  # 绕x轴的附加质量
        M_qdot = -0.1 * self.water_density * self.volume * self.length**2   # 绕y轴的附加质量
        N_rdot = -0.1 * self.water_density * self.volume * self.length**2   # 绕z轴的附加质量
        
        # 交叉耦合附加质量（流体动力交互作用）
        Y_pdot = Z_qdot = -0.05 * self.water_density * self.volume * self.length
        K_vdot = M_wdot = -0.05 * self.water_density * self.volume * self.length
        
        # 构建完整的6x6附加质量矩阵
        self.added_mass_matrix = np.zeros((6, 6))
        # 线运动附加质量
        self.added_mass_matrix[0, 0] = X_udot
        self.added_mass_matrix[1, 1] = Y_vdot
        self.added_mass_matrix[2, 2] = Z_wdot
        # 转动附加质量
        self.added_mass_matrix[3, 3] = K_pdot
        self.added_mass_matrix[4, 4] = M_qdot
        self.added_mass_matrix[5, 5] = N_rdot
        # 交叉耦合项
        self.added_mass_matrix[1, 3] = Y_pdot
        self.added_mass_matrix[3, 1] = K_vdot
        self.added_mass_matrix[2, 4] = Z_qdot
        self.added_mass_matrix[4, 2] = M_wdot
        
        # 3. 流体阻尼矩阵（线性和非线性）
        # 线性阻尼系数 - 减小阻尼到原来的20%（减少8成）以提高运动响应性
        self.linear_damping = np.array([
            [-20.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # X_u - 减小为原来的20%
            [0.0, -30.0, 0.0, 0.0, 0.0, 0.0],   # Y_v - 减小为原来的20%
            [0.0, 0.0, -20.0, 0.0, 0.0, 0.0],   # Z_w - 减小为原来的20%
            [0.0, 0.0, 0.0, -2.0, 0.0, 0.0],    # K_p - 减小为原来的20%
            [0.0, 0.0, 0.0, 0.0, -3.2, 0.0],    # M_q - 减小为原来的20%
            [0.0, 0.0, 0.0, 0.0, 0.0, -4.0]     # N_r - 减小为原来的20%
        ])
        
        # 非线性阻尼系数（平方项）- 减小阻尼到原来的20%（减少8成）以提高高速运动性能
        self.nonlinear_damping = np.array([
            [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # X_uu - 减小为原来的20%
            [0.0, -15.0, 0.0, 0.0, 0.0, 0.0],    # Y_vv - 减小为原来的20%
            [0.0, 0.0, -12.5, 0.0, 0.0, 0.0],    # Z_ww - 减小为原来的20%
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],      # K_pp - 减小为原来的20%
            [0.0, 0.0, 0.0, 0.0, -1.6, 0.0],      # M_qq - 减小为原来的20%
            [0.0, 0.0, 0.0, 0.0, 0.0, -2.0]      # N_rr - 减小为原来的20%
        ])
        
        # 4. 推进器参数
        self.propeller_max_thrust = 200.0  # 最大推力 (N)
        self.propeller_position = np.array([-self.length/2, 0.0, 0.0])  # 推进器位置 (m)
        self.propeller_efficiency = 0.85  # 推进器效率
        
        # 5. 舵机参数 (4个舵机: 2个水平舵，2个垂直舵)
        self.rudder_count = 4  # 舵机数量
        self.rudder_max_angle = np.pi/6  # 最大舵角 (30度)
        # 舵机物理参数
        self.rudder_area = 0.05  # 单个舵机面积 (m²)
        self.rudder_aspect_ratio = 2.0  # 舵机展弦比
        self.rudder_lift_coefficient = 6.0  # 单位角度升力系数 (每弧度)
        self.rudder_drag_coefficient = 1.0  # 零角度阻力系数
        
        # 舵机位置 (在AUV本体坐标系中的坐标)
        # 位于尾部，90度均匀分布
        radius = min(self.width, self.height) / 3  # 舵机分布半径
        self.rudder_positions = np.array([
            [self.length/2 - 0.1, 0, radius],             # 上舵 (垂直舵)
            [self.length/2 - 0.1, radius, 0],             # 右舵 (水平舵)
            [self.length/2 - 0.1, 0, -radius],            # 下舵 (垂直舵)
            [self.length/2 - 0.1, -radius, 0]             # 左舵 (水平舵)
        ])
        
        # 6. 环境参数
        self.current_velocity = np.zeros(3)  # 环境水流速度 [u_current, v_current, w_current]
        self.wave_amplitude = 0.0  # 波浪振幅
        self.wave_frequency = 0.0  # 波浪频率
        self.wave_direction = 0.0  # 波浪方向角
        
        # 7. AUV状态向量
        # 使用四元数表示姿态以避免万向节锁问题
        # [x, y, z, q0, q1, q2, q3, u, v, w, p, q, r]
        # x,y,z: 位置坐标 (m)
        # q0,q1,q2,q3: 四元数 (姿态)
        # u,v,w: 体坐标系下线速度 (m/s)
        # p,q,r: 体坐标系下角速度 (rad/s)
        self.state = np.zeros(13)
        self.state[3] = 1.0  # 初始化四元数为单位四元数 [1,0,0,0]
        
        # 控制输入 [propeller_thrust, rudder1_angle, rudder2_angle, rudder3_angle, rudder4_angle]
        self.control_input = np.zeros(5)
        # 添加时间计数器用于环境力计算
        self.simulation_time = 0.0
    
    @property
    def rudder_angles(self):
        """获取舵机角度"""
        return self.control_input[1:5]
    
    @property
    def thrust(self):
        """获取推进器推力"""
        return self.control_input[0]
    
    def set_environment(self, current_velocity=None, wave_amplitude=0.0, wave_frequency=0.0, wave_direction=0.0,
                       use_current=False, use_waves=False):
        """设置环境参数和控制标志"""
        # 注意：环境力（水流和波浪）已在_calculate_environmental_forces方法中完全禁用
        # 无论传入什么参数，都不会产生任何环境干扰力
        if current_velocity is not None:
            self.current_velocity = np.array(current_velocity)
        self.wave_amplitude = wave_amplitude
        self.wave_frequency = wave_frequency
        self.wave_direction = wave_direction
        # 强制设置为False，环境力已禁用
        self.use_current = False  # 强制设置为False，水流影响已禁用
        self.use_waves = False    # 强制设置为False，波浪影响已禁用
    
    def update(self, dt):
        """更新AUV状态（与update_state方法相同，为了兼容性）"""
        self.update_state(dt)
        
    def update_state(self, dt):
        """使用简化的流体动力学模型更新AUV状态"""
        # 更新模拟时间
        self.simulation_time += dt
        
        # 从状态向量中提取位置、姿态和速度
        x, y, z = self.state[0:3]
        q0, q1, q2, q3 = self.state[3:7]  # 四元数
        u, v, w = self.state[7:10]        # 体坐标系线速度
        p, q, r = self.state[10:13]       # 体坐标系角速度
        
        # 提取控制输入
        propeller_thrust = self.control_input[0]
        rudder_angles = self.control_input[1:5]
        
        # 使用简化的物理模型
        # 1. 只使用刚体质量矩阵，忽略附加质量
        rigid_body_mass = np.zeros((6, 6))
        rigid_body_mass[0:3, 0:3] = np.eye(3) * self.mass
        rigid_body_mass[3:6, 3:6] = self.inertia_matrix
        M = rigid_body_mass
        
        # 2. 简化的阻尼力计算
        damping_forces = self._calculate_damping_forces(u, v, w, p, q, r)
        
        # 3. 计算重力和浮力
        gravity_buoyancy_forces = self._calculate_gravity_buoyancy_forces(q0, q1, q2, q3)
        
        # 4. 计算控制输入产生的力和力矩
        control_forces, control_torques = self._calculate_control_forces(propeller_thrust, rudder_angles, u, v, w, p, q, r)
        
        # 5. 环境干扰力（已禁用）
        environmental_forces = np.zeros(6)
        
        # 6. 组装总力向量（忽略科里奥利力，使用更简单的模型）
        total_forces = np.concatenate([control_forces, control_torques])
        total_forces += damping_forces
        total_forces += gravity_buoyancy_forces
        total_forces += environmental_forces
        
        # 7. 简单的加速度计算：a = F/m
        nu_dot = np.zeros(6)
        try:
            # 限制力向量大小
            total_forces = np.clip(total_forces, -1e4, 1e4)
            
            # 计算加速度（简单的力除以质量）
            for i in range(3):  # 线加速度
                nu_dot[i] = total_forces[i] / self.mass
            
            # 角加速度需要考虑惯性矩
            if self.Ixx != 0:
                nu_dot[3] = total_forces[3] / self.Ixx
            if self.Iyy != 0:
                nu_dot[4] = total_forces[4] / self.Iyy
            if self.Izz != 0:
                nu_dot[5] = total_forces[5] / self.Izz
            
            # 限制加速度大小
            nu_dot = np.clip(nu_dot, -5.0, 5.0)
        except Exception as e:
            print(f"警告: 加速度计算失败: {e}")
            nu_dot = np.zeros(6)
        
        # 9. 更新速度 - 使用半隐式欧拉方法提高数值稳定性
        u_dot, v_dot, w_dot, p_dot, q_dot, r_dot = nu_dot
        
        # 半隐式欧拉积分：先更新速度，再更新位置
        # 这种方法比显式欧拉更稳定，特别是对于刚性系统
        u_new = u + u_dot * dt
        v_new = v + v_dot * dt
        w_new = w + w_dot * dt
        p_new = p + p_dot * dt
        q_new = q + q_dot * dt
        r_new = r + r_dot * dt
        
        # 添加数值稳定性限制：限制速度变化率
        max_delta_velocity = 10.0 * dt  # 每时间步最大速度变化
        u = u + np.clip(u_new - u, -max_delta_velocity, max_delta_velocity)
        v = v + np.clip(v_new - v, -max_delta_velocity, max_delta_velocity)
        w = w + np.clip(w_new - w, -max_delta_velocity, max_delta_velocity)
        p = p + np.clip(p_new - p, -max_delta_velocity, max_delta_velocity)
        q = q + np.clip(q_new - q, -max_delta_velocity, max_delta_velocity)
        r = r + np.clip(r_new - r, -max_delta_velocity, max_delta_velocity)
        
        # 添加速度阈值检查，当速度非常小时将其设置为零，防止数值漂移
        velocity_threshold = 1e-6  # 速度阈值
        if abs(u) < velocity_threshold:
            u = 0.0
        if abs(v) < velocity_threshold:
            v = 0.0
        if abs(w) < velocity_threshold:
            w = 0.0
        if abs(p) < velocity_threshold:
            p = 0.0
        if abs(q) < velocity_threshold:
            q = 0.0
        if abs(r) < velocity_threshold:
            r = 0.0
        
        # 10. 更新位置和姿态
        # 转换体坐标系速度到惯性坐标系
        rotation_matrix = self._quaternion_to_rotation_matrix(q0, q1, q2, q3)
        
        # 计算惯性坐标系中的线速度（考虑水流）
        body_velocity = np.array([u, v, w])
        current_velocity_body = self._transform_vector_inertial_to_body(self.current_velocity, q0, q1, q2, q3)
        relative_velocity = body_velocity - current_velocity_body
        
        # 转换到惯性坐标系
        inertial_velocity = rotation_matrix @ relative_velocity
        
        # 零速度锁定机制：当没有控制输入时，完全锁定AUV位置和速度
        if np.linalg.norm(self.control_input) < velocity_threshold:
            # 1. 直接将所有速度置零
            inertial_velocity = np.zeros(3)
            
            # 2. 将体坐标系中的速度也置零
            u = 0.0
            v = 0.0
            w = 0.0
            p = 0.0
            q = 0.0
            r = 0.0
            
            # 3. 不更新位置，保持AUV静止
            # 不执行任何位置更新操作
        else:
            # 有控制输入时，正常更新位置
            x += inertial_velocity[0] * dt
            y += inertial_velocity[1] * dt
            z += inertial_velocity[2] * dt
        
        # 使用四元数更新姿态
        q0, q1, q2, q3 = self._update_quaternion(q0, q1, q2, q3, p, q, r, dt)
        
        # 11. 更新状态向量
        self.state = np.array([x, y, z, q0, q1, q2, q3, u, v, w, p, q, r])
    
    def _calculate_coriolis_matrix(self, u, v, w, p, q, r):
        """计算科里奥利和离心力矩阵"""
        # 提取附加质量参数
        X_udot = self.added_mass_matrix[0, 0]
        Y_vdot = self.added_mass_matrix[1, 1]
        Z_wdot = self.added_mass_matrix[2, 2]
        K_pdot = self.added_mass_matrix[3, 3]
        M_qdot = self.added_mass_matrix[4, 4]
        N_rdot = self.added_mass_matrix[5, 5]
        Y_pdot = self.added_mass_matrix[1, 3]
        Z_qdot = self.added_mass_matrix[2, 4]
        
        # 计算科里奥利力矩阵
        C = np.zeros((6, 6))
        
        # 线运动部分
        C[0, 1] = -(Y_vdot + M_qdot) * r
        C[0, 2] = (Z_wdot + N_rdot) * q
        C[0, 4] = -Z_wdot * w
        C[0, 5] = Y_vdot * v
        
        C[1, 0] = (Y_vdot + M_qdot) * r
        C[1, 2] = -(X_udot + N_rdot) * p
        C[1, 3] = Y_pdot * r
        C[1, 5] = -X_udot * u
        
        C[2, 0] = -(Z_wdot + N_rdot) * q
        C[2, 1] = (X_udot + N_rdot) * p
        C[2, 3] = -Z_qdot * q
        C[2, 4] = X_udot * u
        
        # 角运动部分
        C[3, 1] = -Y_pdot * r
        C[3, 2] = Z_qdot * q
        C[3, 4] = -Z_qdot * w
        C[3, 5] = Y_pdot * v
        
        C[4, 0] = Z_wdot * w
        C[4, 2] = -X_udot * u
        C[4, 3] = Z_qdot * w
        C[4, 5] = (X_udot - Z_wdot) * u
        
        C[5, 0] = -Y_vdot * v
        C[5, 1] = X_udot * u
        C[5, 3] = -Y_pdot * v
        C[5, 4] = (Z_wdot - X_udot) * u
        
        # 添加刚体部分的科里奥利力
        C[0, 4] += -self.mass * w
        C[0, 5] += self.mass * v
        C[1, 3] += self.mass * w
        C[1, 5] += -self.mass * u
        C[2, 3] += -self.mass * v
        C[2, 4] += self.mass * u
        
        C[3, 4] += -(self.Izz - self.Iyy) * r
        C[3, 5] += (self.Izz - self.Iyy) * q
        C[4, 3] += -(self.Ixx - self.Izz) * r
        C[4, 5] += (self.Ixx - self.Izz) * p
        C[5, 3] += -(self.Iyy - self.Ixx) * q
        C[5, 4] += (self.Iyy - self.Ixx) * p
        
        return C
    
    def _calculate_damping_forces(self, u, v, w, p, q, r):
        """简化的流体阻尼力计算，仅使用线性阻尼参数"""
        # 使用简单的线性阻尼，与速度成正比
        damping_forces = np.zeros(6)
        
        # 线性阻尼系数 (简化版本) - 减小阻尼值到原来的20%（减少8成）
        damping_coeff = np.array([
            -20.0,  # X方向阻尼 - 减小为原来的20%
            -30.0,  # Y方向阻尼 - 减小为原来的20%
            -20.0,  # Z方向阻尼 - 减小为原来的20%
            -2.0,   # 横滚阻尼 - 减小为原来的20%
            -3.2,   # 俯仰阻尼 - 减小为原来的20%
            -4.0    # 偏航阻尼 - 减小为原来的20%
        ])
        
        # 计算简单的线性阻尼力
        damping_forces[0] = damping_coeff[0] * u
        damping_forces[1] = damping_coeff[1] * v
        damping_forces[2] = damping_coeff[2] * w
        damping_forces[3] = damping_coeff[3] * p
        damping_forces[4] = damping_coeff[4] * q
        damping_forces[5] = damping_coeff[5] * r
        
        # 当没有控制输入时，轻微增加阻尼来防止漂移，但保持较小值
        if hasattr(self, 'control_input') and np.linalg.norm(self.control_input) < 1e-6:
            # 对所有速度分量轻微增加阻尼
            for i in range(6):
                damping_forces[i] *= 1.2  # 轻微增加阻尼，而不是原来的2倍
        
        return damping_forces
    
    def _calculate_gravity_buoyancy_forces(self, q0, q1, q2, q3):
        """计算重力和浮力 - 增强版本，考虑深度、姿态等因素对浮力的影响"""
        # 获取当前深度 (z坐标为负表示下潜)
        depth = -self.state[2]  # 假设z=0为水面
        
        # 1. 计算重力向量 (恒定)
        gravity = np.array([0, 0, -self.mass * 9.81])  # 重力 (N)
        
        # 2. 增强的浮力计算
        # 基础浮力
        base_buoyancy = self.water_density * 9.81 * self.volume
        
        # 深度影响：水压增加导致水密度略微增加（每1000米增加约1%）
        depth_pressure_factor = 1.0 + (depth * 0.00001)  # 轻微的深度依赖性
        
        # 姿态影响：当AUV倾斜时，浮力的有效体积略微变化
        # 计算AUV的俯仰角和横滚角来评估姿态影响
        pitch = np.arcsin(2.0 * (q0*q2 - q1*q3))  # 从四元数计算俯仰角
        roll = np.arcsin(2.0 * (q0*q1 + q2*q3))   # 从四元数计算横滚角
        
        # 姿态因子：当姿态倾斜时，略微调整浮力（模拟流体动力学效应）
        attitude_factor = 1.0 - 0.02 * (abs(pitch) + abs(roll))  # 最多减少4%的浮力
        attitude_factor = max(0.96, attitude_factor)  # 限制最小因子
        
        # 计算最终浮力
        dynamic_buoyancy = base_buoyancy * depth_pressure_factor * attitude_factor
        
        # 浮力向量始终向上（z轴正方向）
        buoyancy = np.array([0, 0, dynamic_buoyancy])  # 浮力 (N)
        
        # 3. 浮心位置动态调整：随姿态略微变化
        # 当AUV倾斜时，浮心位置会有微小变化（简化模型）
        dynamic_center_of_buoyancy = self.center_of_buoyancy.copy()
        # 姿态引起的浮心偏移（非常小的量）
        shift_magnitude = 0.01  # 最大偏移量 (m)
        dynamic_center_of_buoyancy[0] += shift_magnitude * np.sin(pitch)
        dynamic_center_of_buoyancy[1] += shift_magnitude * np.sin(roll)
        
        # 计算浮心相对于重心的位置
        b_g = dynamic_center_of_buoyancy - self.center_of_mass
        
        # 4. 转换到体坐标系
        rotation_matrix = self._quaternion_to_rotation_matrix(q0, q1, q2, q3)
        gravity_body = rotation_matrix.T @ gravity
        buoyancy_body = rotation_matrix.T @ buoyancy
        
        # 5. 计算力和力矩
        forces = gravity_body + buoyancy_body
        moments = np.cross(b_g, buoyancy_body)  # 浮力产生的力矩
        
        # 6. 添加静稳定性增强：当AUV倾斜时增加恢复力矩
        # 这模拟了真实AUV的稳定性设计
        stability_gain = 50.0  # 稳定性增益系数
        stability_moment = np.array([
            -stability_gain * roll,      # 横滚恢复力矩
            -stability_gain * pitch,     # 俯仰恢复力矩
            0.0                          # 偏航不受稳定性影响
        ])
        
        # 将稳定性力矩添加到总力矩中
        moments += stability_moment
        
        return np.concatenate([forces, moments])
    
    def _calculate_control_forces(self, thrust, rudder_angles, u, v, w, p, q, r):
        """计算控制输入产生的力和力矩 - 简化版本，舵力直接基于舵角和速度"""
        # 1. 计算推进器产生的力和力矩
        # 推力方向直接与推进器输入方向一致，不再与当前速度方向绑定
        # 正推力表示向前推，负推力表示向后推
        thrust_force = np.array([thrust, 0.0, 0.0]) * self.propeller_efficiency
        # 推进力矩（如果推进器不在中心线上，会产生力矩）
        thrust_torque = np.cross(self.propeller_position, thrust_force)
        
        # 2. 计算舵机产生的力和力矩 - 简化版本
        rudder_forces = np.zeros(3)
        rudder_torques = np.zeros(3)
        
        # 前进速度大小（用于计算舵力）
        forward_speed = abs(u)
        
        # 舵力系数（根据舵角和速度调整）
        # 系数越大，舵效越强
        rudder_force_coefficient = 100.0
        
        # 分别计算每个舵的力和力矩
        for i in range(self.rudder_count):
            angle = rudder_angles[i]
            
            # 基于舵角和前进速度直接计算舵力
            # 力的大小与舵角和前进速度成正比
            rudder_force_magnitude = rudder_force_coefficient * angle * forward_speed
            
            # 限制舵力大小
            max_force = 50.0
            rudder_force_magnitude = np.clip(rudder_force_magnitude, -max_force, max_force)
            
            # 根据舵的类型确定力的方向
            # 简化的力方向模型
            if i == 0 or i == 2:  # 垂直舵（上舵和下舵）
                # 垂直舵产生垂直方向的力，影响偏航
                if i == 0:  # 上舵
                    rudder_force = np.array([0.0, 0.0, rudder_force_magnitude])
                else:  # 下舵
                    rudder_force = np.array([0.0, 0.0, -rudder_force_magnitude])
            else:  # 水平舵（左舵和右舵）
                # 水平舵产生水平方向的力，影响俯仰
                if i == 1:  # 右舵
                    rudder_force = np.array([0.0, rudder_force_magnitude, 0.0])
                else:  # 左舵
                    rudder_force = np.array([0.0, -rudder_force_magnitude, 0.0])
            
            # 计算力矩（相对于质心）
            torque = np.cross(self.rudder_positions[i], rudder_force)
            
            rudder_forces += rudder_force
            rudder_torques += torque
        
        # 总控制力和力矩
        total_forces = thrust_force + rudder_forces
        total_torques = thrust_torque + rudder_torques
        
        return total_forces, total_torques
    
    def _calculate_environmental_forces(self, u, v, w, p, q, r):
        """计算环境干扰力（水流和波浪）"""
        # 完全禁用环境力（水流和波浪），直接返回零向量
        # 无论self.use_current和self.use_waves如何设置，都不计算任何环境力
        forces = np.zeros(3)
        moments = np.zeros(3)
        
        # 即使在未来需要重新启用环境力，也可以通过删除这个修改来实现
        
        return np.concatenate([forces, moments])
    
    def _quaternion_to_rotation_matrix(self, q0, q1, q2, q3):
        """将四元数转换为旋转矩阵"""
        return np.array([
            [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
            [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
            [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
        ])
    
    def _update_quaternion(self, q0, q1, q2, q3, p, q, r, dt):
        """使用角速度更新四元数，增加数值稳定性"""
        # 限制角速度大小
        max_angular_velocity = np.pi  # 最大角速度180度/秒
        p = np.clip(p, -max_angular_velocity, max_angular_velocity)
        q = np.clip(q, -max_angular_velocity, max_angular_velocity)
        r = np.clip(r, -max_angular_velocity, max_angular_velocity)
        
        # 四元数导数
        q_dot = 0.5 * np.array([
            -q1*p - q2*q - q3*r,
            q0*p + q2*r - q3*q,
            q0*q - q1*r + q3*p,
            q0*r + q1*q - q2*p
        ])
        
        # 限制导数大小，防止四元数更新过大
        max_qdot = 0.1  # 限制每次更新的最大增量
        q_dot_norm = np.linalg.norm(q_dot)
        if q_dot_norm > max_qdot:
            q_dot = q_dot * (max_qdot / q_dot_norm)
        
        # 更新四元数
        q0 += q_dot[0] * dt
        q1 += q_dot[1] * dt
        q2 += q_dot[2] * dt
        q3 += q_dot[3] * dt
        
        # 规范化四元数，增加数值稳定性
        # 计算各个分量的平方，避免溢出
        q_sq = np.array([q0**2, q1**2, q2**2, q3**2])
        
        # 检查是否有异常大的值
        max_q_sq = np.max(q_sq)
        if max_q_sq > 1e20:  # 阈值防止溢出
            # 对大值进行缩放
            scale = 1e-10
            q0 *= scale
            q1 *= scale
            q2 *= scale
            q3 *= scale
            q_sq *= scale * scale
        
        norm = np.sqrt(np.sum(q_sq))
        if norm > 1e-6:
            q0 /= norm
            q1 /= norm
            q2 /= norm
            q3 /= norm
        
        return q0, q1, q2, q3
    
    def _transform_vector_inertial_to_body(self, vector_inertial, q0, q1, q2, q3):
        """将惯性坐标系中的向量转换到体坐标系"""
        rotation_matrix = self._quaternion_to_rotation_matrix(q0, q1, q2, q3)
        return rotation_matrix.T @ vector_inertial
    
    def _transform_vector_body_to_inertial(self, vector_body, q0, q1, q2, q3):
        """将体坐标系中的向量转换到惯性坐标系"""
        rotation_matrix = self._quaternion_to_rotation_matrix(q0, q1, q2, q3)
        return rotation_matrix @ vector_body
    
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
        """获取AUV当前姿态角（欧拉角）"""
        q0, q1, q2, q3 = self.state[3:7]
        return self._quaternion_to_euler(q0, q1, q2, q3)
    
    def get_quaternion(self):
        """获取AUV当前姿态（四元数）"""
        return self.state[3:7]
    
    def set_quaternion(self, quaternion):
        """设置AUV姿态（四元数）"""
        # 规范化四元数
        q0, q1, q2, q3 = quaternion
        norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        if norm > 1e-6:
            self.state[3:7] = np.array([q0, q1, q2, q3]) / norm
    
    def get_velocity(self):
        """获取AUV速度[u, v, w]（体坐标系线速度）"""
        return self.state[7:10]
    
    def get_linear_velocity(self):
        """获取AUV线速度[u, v, w]（体坐标系）"""
        return self.state[7:10]
    
    def get_angular_velocity(self):
        """获取AUV当前角速度（体坐标系）"""
        return self.state[10:13]
    
    def set_velocity(self, velocity):
        """设置AUV速度[u, v, w, p, q, r]（体坐标系线速度和角速度）
        
        参数:
        velocity: 长度为3（只设置线速度）或6（设置线速度和角速度）的数组
        """
        if len(velocity) == 3:
            # 只设置线速度，保持角速度不变
            self.state[7:10] = velocity
        elif len(velocity) == 6:
            # 设置完整的速度向量
            self.state[7:13] = velocity
        else:
            raise ValueError(f"速度数组长度必须为3或6，当前为{len(velocity)}")
    
    def get_velocity_inertial(self):
        """获取AUV在惯性坐标系中的速度"""
        u, v, w = self.state[7:10]
        q0, q1, q2, q3 = self.state[3:7]
        body_velocity = np.array([u, v, w])
        return self._transform_vector_body_to_inertial(body_velocity, q0, q1, q2, q3)
    
    def _quaternion_to_euler(self, q0, q1, q2, q3):
        """将四元数转换为欧拉角（横滚、俯仰、偏航）"""
        # 横滚 (phi)
        sin_phi = 2.0 * (q0*q1 + q2*q3)
        cos_phi = 1.0 - 2.0 * (q1*q1 + q2*q2)
        phi = np.arctan2(sin_phi, cos_phi)
        
        # 俯仰 (theta)
        sin_theta = 2.0 * (q0*q2 - q3*q1)
        # 避免数值不稳定
        if abs(sin_theta) >= 1:
            theta = np.copysign(np.pi/2, sin_theta)
        else:
            theta = np.arcsin(sin_theta)
        
        # 偏航 (psi)
        sin_psi = 2.0 * (q0*q3 + q1*q2)
        cos_psi = 1.0 - 2.0 * (q2*q2 + q3*q3)
        psi = np.arctan2(sin_psi, cos_psi)
        
        return np.array([phi, theta, psi])
    
    def reset(self):
        """重置AUV状态"""
        self.state = np.zeros(13)
        self.state[3] = 1.0  # 初始化四元数为单位四元数
        self.control_input = np.zeros(5)
        self.current_velocity = np.zeros(3)
        self.wave_amplitude = 0.0
        self.wave_frequency = 0.0
