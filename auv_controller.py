import numpy as np

class AUVController:
    def __init__(self, auv_model):
        self.auv_model = auv_model
        
        # 控制参数
        self.thrust = 0.0
        self.rudder_angles = np.zeros(4)
        
        # PID控制器参数
        self.pid_params = {
            'depth': {'kp': 10.0, 'ki': 0.1, 'kd': 2.0},
            'heading': {'kp': 5.0, 'ki': 0.05, 'kd': 1.0},
            'pitch': {'kp': 8.0, 'ki': 0.1, 'kd': 1.5},
            'roll': {'kp': 6.0, 'ki': 0.05, 'kd': 1.0}
        }
        
        # PID控制器状态
        self.pid_errors = {
            'depth': {'p': 0.0, 'i': 0.0, 'd': 0.0, 'prev': 0.0},
            'heading': {'p': 0.0, 'i': 0.0, 'd': 0.0, 'prev': 0.0},
            'pitch': {'p': 0.0, 'i': 0.0, 'd': 0.0, 'prev': 0.0},
            'roll': {'p': 0.0, 'i': 0.0, 'd': 0.0, 'prev': 0.0}
        }
        
        # 目标值
        self.targets = {
            'depth': 0.0,  # 目标深度
            'heading': 0.0,  # 目标航向
            'pitch': 0.0,  # 目标俯仰角
            'roll': 0.0  # 目标横滚角
        }
        
        # 键盘控制状态
        self.key_state = {
            'thrust_forward': False,
            'thrust_backward': False,
            'turn_left': False,
            'turn_right': False,
            'pitch_up': False,
            'pitch_down': False,
            'roll_left': False,
            'roll_right': False,
            'depth_up': False,
            'depth_down': False,
            # 独立舵板控制
            'rudder1_up': False,
            'rudder1_down': False,
            'rudder2_up': False,
            'rudder2_down': False,
            'rudder3_up': False,
            'rudder3_down': False,
            'rudder4_up': False,
            'rudder4_down': False
        }
    
    def update(self, dt):
        """更新控制器状态"""
        # 基于键盘输入更新控制指令
        self._update_from_keyboard()
        
        # 使用PID控制器计算控制量
        self._update_pid_controllers(dt)
        
        # 将控制量应用到AUV模型
        self.auv_model.set_control_input(self.thrust, self.rudder_angles)
    
    def _update_from_keyboard(self):
        """根据键盘输入更新控制状态"""
        # 重置推力
        self.thrust = 0.0
        
        # 前进/后退控制
        if self.key_state['thrust_forward']:
            self.thrust = self.auv_model.propeller_max_thrust * 0.8  # 前进使用正推力
        elif self.key_state['thrust_backward']:
            self.thrust = -self.auv_model.propeller_max_thrust * 0.3  # 后退使用负推力
        
        # 检查是否有独立舵板控制按键被按下
        has_independent_rudder_control = any([
            self.key_state['rudder1_up'], self.key_state['rudder1_down'],
            self.key_state['rudder2_up'], self.key_state['rudder2_down'],
            self.key_state['rudder3_up'], self.key_state['rudder3_down'],
            self.key_state['rudder4_up'], self.key_state['rudder4_down']
        ])
        
        # 独立舵板控制（最高优先级）
        if has_independent_rudder_control:
            # 重置所有舵角
            self.rudder_angles = np.zeros(4)
            
            # 舵板1控制
            if self.key_state['rudder1_up']:
                self.rudder_angles[0] += self.auv_model.rudder_max_angle * 0.8
            elif self.key_state['rudder1_down']:
                self.rudder_angles[0] -= self.auv_model.rudder_max_angle * 0.8
            
            # 舵板2控制
            if self.key_state['rudder2_up']:
                self.rudder_angles[1] += self.auv_model.rudder_max_angle * 0.8
            elif self.key_state['rudder2_down']:
                self.rudder_angles[1] -= self.auv_model.rudder_max_angle * 0.8
            
            # 舵板3控制
            if self.key_state['rudder3_up']:
                self.rudder_angles[2] += self.auv_model.rudder_max_angle * 0.8
            elif self.key_state['rudder3_down']:
                self.rudder_angles[2] -= self.auv_model.rudder_max_angle * 0.8
            
            # 舵板4控制
            if self.key_state['rudder4_up']:
                self.rudder_angles[3] += self.auv_model.rudder_max_angle * 0.8
            elif self.key_state['rudder4_down']:
                self.rudder_angles[3] -= self.auv_model.rudder_max_angle * 0.8
        
        # 传统的组合控制（次优先级）
        elif any([self.key_state['turn_left'], self.key_state['turn_right'],
                self.key_state['pitch_up'], self.key_state['pitch_down'],
                self.key_state['roll_left'], self.key_state['roll_right']]):
            # 转向控制（偏航）
            turn_angle = 0.0
            if self.key_state['turn_left']:
                turn_angle -= self.auv_model.rudder_max_angle * 0.8  # 左转向：上垂直舵向左偏，下垂直舵向右偏
            elif self.key_state['turn_right']:
                turn_angle += self.auv_model.rudder_max_angle * 0.8  # 右转向：上垂直舵向右偏，下垂直舵向左偏
            
            # 俯仰控制
            pitch_angle = 0.0
            if self.key_state['pitch_up']:
                pitch_angle += self.auv_model.rudder_max_angle * 0.6
            elif self.key_state['pitch_down']:
                pitch_angle -= self.auv_model.rudder_max_angle * 0.6
            
            # 横滚控制
            roll_angle = 0.0
            if self.key_state['roll_left']:
                roll_angle += self.auv_model.rudder_max_angle * 0.6
            elif self.key_state['roll_right']:
                roll_angle -= self.auv_model.rudder_max_angle * 0.6
            
            # 设置舵角
            # 垂直舵（舵板0和2）控制转向
            # 水平舵（舵板1和3）控制俯仰和横滚
            # 左转向：上垂直舵向左偏，下垂直舵向右偏
            # 右转向：上垂直舵向右偏，下垂直舵向左偏
            # 左横滚：右水平舵上偏，左水平舵下偏
            # 右横滚：右水平舵下偏，左水平舵上偏
            self.rudder_angles[0] = turn_angle                # 上垂直舵 - 控制转向
            self.rudder_angles[1] = pitch_angle + roll_angle  # 右水平舵 - 控制俯仰和横滚
            self.rudder_angles[2] = -turn_angle               # 下垂直舵 - 控制转向
            self.rudder_angles[3] = pitch_angle - roll_angle  # 左水平舵 - 控制俯仰和横滚
        
        # 限制舵角范围
        for i in range(4):
            self.rudder_angles[i] = np.clip(
                self.rudder_angles[i], 
                -self.auv_model.rudder_max_angle, 
                self.auv_model.rudder_max_angle
            )
    
    def _update_pid_controllers(self, dt):
        """更新PID控制器计算"""
        current_depth = -self.auv_model.get_position()[2]  # z轴向下为正
        current_heading = self.auv_model.get_orientation()[2]  # psi角
        current_pitch = self.auv_model.get_orientation()[1]  # theta角
        current_roll = self.auv_model.get_orientation()[0]  # phi角
        
        # 计算深度PID
        depth_error = self.targets['depth'] - current_depth
        self._update_pid('depth', depth_error, dt)
        depth_control = self._calculate_pid_output('depth')
        
        # 计算航向PID
        heading_error = self._normalize_angle(self.targets['heading'] - current_heading)
        self._update_pid('heading', heading_error, dt)
        heading_control = self._calculate_pid_output('heading')
        
        # 计算俯仰PID
        pitch_error = self.targets['pitch'] - current_pitch
        self._update_pid('pitch', pitch_error, dt)
        pitch_control = self._calculate_pid_output('pitch')
        
        # 计算横滚PID
        roll_error = self.targets['roll'] - current_roll
        self._update_pid('roll', roll_error, dt)
        roll_control = self._calculate_pid_output('roll')
        
        # 无键盘控制时，舵角归零
        # 移除PID控制器在无键盘操作时自动设置舵角的逻辑
        # 只有当有键盘控制时，才考虑PID控制
        pass
    
    def _update_pid(self, controller_name, error, dt):
        """更新单个PID控制器的误差值"""
        self.pid_errors[controller_name]['p'] = error
        self.pid_errors[controller_name]['i'] += error * dt
        self.pid_errors[controller_name]['d'] = (error - self.pid_errors[controller_name]['prev']) / dt
        self.pid_errors[controller_name]['prev'] = error
        
        # 积分饱和限制
        max_integral = 10.0
        self.pid_errors[controller_name]['i'] = np.clip(
            self.pid_errors[controller_name]['i'], 
            -max_integral, 
            max_integral
        )
    
    def _calculate_pid_output(self, controller_name):
        """计算PID控制器输出"""
        p = self.pid_params[controller_name]['kp'] * self.pid_errors[controller_name]['p']
        i = self.pid_params[controller_name]['ki'] * self.pid_errors[controller_name]['i']
        d = self.pid_params[controller_name]['kd'] * self.pid_errors[controller_name]['d']
        
        output = p + i + d
        
        # 限制输出范围
        max_output = self.auv_model.rudder_max_angle
        return np.clip(output, -max_output, max_output)
    
    def set_key_state(self, key, pressed):
        """设置键盘按键状态"""
        key_map = {
            # 基本控制按键
            'w': 'thrust_forward',
            's': 'thrust_backward',
            'a': 'turn_left',
            'd': 'turn_right',
            'up': 'pitch_up',
            'down': 'pitch_down',
            'left': 'roll_left',
            'right': 'roll_right',
            'q': 'depth_up',
            'e': 'depth_down',
            # 独立舵板控制按键
            '1': 'rudder1_up',
            '!': 'rudder1_down',  # Shift+1
            '2': 'rudder2_up',
            '@': 'rudder2_down',  # Shift+2
            '3': 'rudder3_up',
            '#': 'rudder3_down',  # Shift+3
            '4': 'rudder4_up',
            '$': 'rudder4_down'   # Shift+4
        }
        
        if key in key_map:
            self.key_state[key_map[key]] = pressed
    
    def set_target_depth(self, depth):
        """设置目标深度"""
        self.targets['depth'] = depth
    
    def set_target_heading(self, heading):
        """设置目标航向"""
        self.targets['heading'] = self._normalize_angle(heading)
    
    def set_target_pitch(self, pitch):
        """设置目标俯仰角"""
        self.targets['pitch'] = np.clip(pitch, -np.pi/6, np.pi/6)  # 限制在±30度
    
    def set_target_roll(self, roll):
        """设置目标横滚角"""
        self.targets['roll'] = np.clip(roll, -np.pi/6, np.pi/6)  # 限制在±30度
    
    def get_control_state(self):
        """获取当前控制状态"""
        return {
            'thrust': self.thrust,
            'rudder_angles': self.rudder_angles.copy(),
            'targets': self.targets.copy(),
            'current_depth': -self.auv_model.get_position()[2],
            'current_heading': self.auv_model.get_orientation()[2],
            'current_pitch': self.auv_model.get_orientation()[1],
            'current_roll': self.auv_model.get_orientation()[0]
        }
    
    def _normalize_angle(self, angle):
        """将角度规范化到 [-pi, pi] 范围内"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def reset(self):
        """重置控制器状态"""
        self.thrust = 0.0
        self.rudder_angles = np.zeros(4)
        
        # 重置PID状态
        for controller in self.pid_errors:
            self.pid_errors[controller] = {'p': 0.0, 'i': 0.0, 'd': 0.0, 'prev': 0.0}
        
        # 重置键盘状态
        for key in self.key_state:
            self.key_state[key] = False
