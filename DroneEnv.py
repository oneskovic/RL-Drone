import gym
import numpy as np
from gym import spaces
import pygame
from time import sleep
from copy import deepcopy
import progressbar

delta_t = 0.1


class Force:
    def __init__(self, positon_vec, force_vec):
        self.position_vec = positon_vec
        self.force_vec = force_vec


class Drone:
    def __init__(self, position_vec, length, mass, angle):
        self.position_vec = position_vec
        self.length = length
        self.angle = angle
        self.mass = mass
        self.velocity_vec = np.zeros((2,))
        self.angular_velocity = 0.0  # In radians

    def __net_force(self, forces):
        net_force = np.zeros((2,))
        for force in forces:
            net_force += force.force_vec
        return net_force

    def __net_torque(self, forces):
        net_torque = 0.0
        for force in forces:
            net_torque += np.cross((force.position_vec - self.position_vec), force.force_vec)
        return net_torque

    def apply_forces(self, forces):
        net_force = self.__net_force(forces)
        net_torque = self.__net_torque(forces)

        self.velocity_vec += net_force / self.mass * delta_t
        rod_moment_of_intertia = self.mass * self.length ** 2 / 12.0
        self.angular_velocity += net_torque / rod_moment_of_intertia * delta_t

    def move(self):
        self.position_vec += self.velocity_vec * delta_t
        self.angle += self.angular_velocity * delta_t
        if self.angle > 2 * np.pi:
            self.angle -= 2 * np.pi
        if self.angle < -2 * np.pi:
            self.angle += 2 * np.pi

    def get_endpoints(self):

        bx = self.length / 2.0 * np.cos(self.angle)
        by = self.length / 2.0 * np.sin(self.angle)
        B = np.array([bx, by]) + self.position_vec
        A = 2 * self.position_vec - B
        return A, B


class DroneEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DroneEnv, self).__init__()
        self.env_min_x = -250.0
        self.env_max_x = 250.0
        self.env_min_y = -250.0
        self.env_max_y = 250.0
        self.max_rotor_force = 75.0

        self.action_space = spaces.Box(low=0.0, high=self.max_rotor_force, shape=(2,), dtype=np.float32)
        # Observation: [dx, dy, vx, vy, w, angle]
        # dx, dy: distance from starting point
        # vx, vy: velocity in directions x and y
        # w: drone angular velocity
        # angle: current drone angle
        # tx, ty: target position x and y
        self.observation_space = spaces.Box(low=self.env_min_x, high=self.env_max_x,
                                            shape=(8,), dtype=np.float32)
        self.reset()

    def __random_point_on_drone(self):
        A, B = self.drone.get_endpoints()

        # Pick which side to generate the wind on
        offset_vec = A - self.drone.position_vec
        if np.random.randint(0,2) == 0:
            offset_vec = B - self.drone.position_vec

        # Scale the offset vector
        offset_scale = np.random.uniform(0.0, 1.0)
        offset_vec *= offset_scale
        position = self.drone.position_vec + offset_vec
        return position

    def __generate_random_wind(self):
        wind_force_x = np.random.uniform(-self.wind_max_magnitude, self.wind_max_magnitude)
        wind_force_y = np.random.uniform(-np.sqrt(self.wind_max_magnitude ** 2 - wind_force_x ** 2),
                                         np.sqrt(self.wind_max_magnitude ** 2 - wind_force_x ** 2))

        wind = Force(self.__random_point_on_drone(), np.array([wind_force_x, wind_force_y]))

        return wind

    # Returns an implicit line formula in form: Ax + Bx + C
    def __line_equation_given_two_points(self, point_a, point_b):
        Ax = point_a[0]
        Ay = point_a[1]
        Bx = point_b[0]
        By = point_b[1]
        return [By - Ay, Ax - Bx, By * (Bx - Ax) - Bx * (By - Ay)]

    # Finds intersection of two lines given two lines in implicit form
    # Returns vector [x,y] - the point of intersection
    def __intersection_of_lines(self, line1, line2):
        A = [[line1[0], line1[1]],
             [line2[0], line2[1]]]
        B = [[-line1[2]],
             [-line2[2]]]
        A = np.array(A, dtype=np.float32)
        B = np.array(B, dtype=np.float32)
        if np.linalg.matrix_rank(A) != A.shape[0]:
            return None
        return np.linalg.solve(A, B).flat

    def __translate_wind_to_drone(self):
        if np.linalg.norm(self.current_wind.force_vec) > 0:
            A, B = self.drone.get_endpoints()
            drone_line = self.__line_equation_given_two_points(A, B)
            wind_line = self.__line_equation_given_two_points(self.current_wind.position_vec,
                                                              self.current_wind.position_vec + self.current_wind.force_vec)
            intersection = self.__intersection_of_lines(drone_line, wind_line)
            if intersection is not None and min(A[0], B[0]) <= intersection[0] <= max(A[0], B[0]):
                # Check which side the wind is on
                Ax = A - intersection
                Bx = B - intersection

                closer_point = B
                # Wind is closer to A
                if np.linalg.norm(Ax) < np.linalg.norm(Bx):
                    closer_point = A
                else:
                    closer_point = B

                translation_vec = (intersection - closer_point) + (closer_point - self.current_wind.position_vec)
                self.current_wind.position_vec += translation_vec
            else:
                self.current_wind = Force(self.drone.position_vec, np.array([0.0, 0.0]))
                self.current_wind_remaining_time = 0

    def __generate_random_target(self):
        rand_x = np.random.uniform(self.env_min_x+self.drone.length, self.env_max_x-self.drone.length)
        rand_y = np.random.uniform(self.env_min_y+self.drone.length, self.env_max_y-self.drone.length)
        return np.array([rand_x, rand_y])

    def __get_rotor_force(self, rotor_point, force_magnitude, orientation):
        center = self.drone.position_vec
        fx = -(rotor_point[1] - center[1])
        fy = rotor_point[0] - center[0]
        force_vec = np.array([fx, fy], dtype=np.float32)
        force_vec = force_vec / np.linalg.norm(force_vec)
        force_vec *= force_magnitude
        force_vec *= orientation

        return Force(rotor_point, force_vec)

    def __minmax_normalize(self, min_value, max_value, x):
        return (x - min_value) / (max_value - min_value)

    def __get_current_state(self):
        position = self.drone.position_vec
        target_pos = self.target_position
        velocity = self.drone.velocity_vec
        angular_velocity = self.drone.angular_velocity
        angle = self.drone.angle
        state = np.array([position[0], position[1], velocity[0], velocity[1], angular_velocity, angle, target_pos[0], target_pos[1]],
                         dtype=np.float32)
        return state

    def step(self, action):

        # region Generate, check and translate wind
        # Check if new wind should be generated
        if self.current_wind_remaining_time == 0:
            self.current_wind = Force(self.drone.position_vec, np.array([0.0, 0.0]))
            r = np.random.uniform(0, 1)
            if r <= self.wind_probability:
                self.current_wind = self.__generate_random_wind()
                self.current_wind_remaining_time = np.random.randint(self.wind_duration[0], self.wind_duration[1])
        else:
            self.current_wind_remaining_time -= 1
        self.__translate_wind_to_drone()
        # endregion

        # region Generate other forces, apply them and move the drone
        gravity = Force(self.drone.position_vec, np.array([0.0, 9.81 * self.drone.mass]))
        A, B = self.drone.get_endpoints()

        force1 = self.__get_rotor_force(A, action[0], 1)
        force2 = self.__get_rotor_force(B, action[1], -1)

        self.current_forces = [gravity, force1, force2, self.current_wind]
        self.drone.apply_forces([gravity, force1, force2, self.current_wind])
        self.drone.move()
        # endregion

        # Compute the reward
        current_position = self.drone.position_vec
        distance_from_target = np.linalg.norm(current_position - self.target_position)
        distance_from_target_normalized = self.__minmax_normalize(0, np.sqrt(2) * 2 * self.env_max_x, distance_from_target)

        reward = 1/(distance_from_target + 1e-10)
        #reward = -distance_from_target_normalized/1.0
        #reward = self.__calculate_reward(distance_from_target)
        observation = self.__get_current_state()

        done = False
        if self.drone.position_vec[0] > self.env_max_x or self.drone.position_vec[0] < self.env_min_x or \
                self.drone.position_vec[1] > self.env_max_y or self.drone.position_vec[1] < self.env_min_y:
            done = True
            reward = -4200.0

        if self.step_count > 1000:
            done = True

        if done:
            pygame.display.quit()
            pygame.quit()
            self.render_started = False

        self.step_count += 1
        return observation, reward, done, {}

    def reset(self):
        drone_length = 40.0
        drone_mass = 5.0
        drone_angle = 0.0
        self.drone = Drone(np.array([0.0, 0.0]), drone_length, drone_mass, drone_angle)
        self.step_count = 0
        self.render_started = False

        self.wind_probability = 0.1
        self.wind_duration = np.array([10, 100], dtype=np.int32)
        self.wind_max_magnitude = (self.max_rotor_force - drone_mass*9.81) / 2.0

        self.current_wind = Force(self.drone.position_vec, np.array([0.0, 0.0]))
        self.current_wind_remaining_time = 0
        self.target_position = self.__generate_random_target()
        return self.__get_current_state()

    def render(self, mode='human'):
        if not self.render_started:
            self.render_started = True
            pygame.init()
            pygame.display.set_caption("minimal program")
            self.pygame_screen = pygame.display.set_mode((500, 500))
            self.episode_progressbar = progressbar.ProgressBar(0, 1002, redirect_stdout=True)

        pygame.event.pump()
        self.pygame_screen.fill(pygame.Color('Black'))
        A, B = self.drone.get_endpoints()
        A += np.array([250.0, 250.0])
        B += np.array([250.0, 250.0])

        for f in self.current_forces:
            force = deepcopy(f)
            force_start_pos = force.position_vec
            force_end_pos = force_start_pos + force.force_vec
            force_start_pos += np.array([250.0, 250.0])
            force_end_pos += np.array([250.0, 250.0])

            pygame.draw.line(self.pygame_screen, pygame.Color('Blue'), force_start_pos, force_end_pos, 2)

        pygame.draw.line(self.pygame_screen, pygame.Color('Red'), A, B, 2)
        target_pos = self.target_position + np.array([250.0, 250.0])
        pygame.draw.circle(self.pygame_screen, pygame.Color('Green'), target_pos.tolist(), 5.0)
        pygame.display.flip()
        self.episode_progressbar.update(self.step_count)
        sleep(0.01)

    def close(self):
        if self.render_started:
            pygame.display.quit()
            pygame.quit()
            self.render_started = False
